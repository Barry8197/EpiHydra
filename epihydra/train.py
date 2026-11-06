"""
EpiHydra training utilities.

Functions
---------
build_multiclock_dataloaders : Align, split, wrap methylation data into DataLoaders for multiple clocks
evaluate_multiclock : Run inference, return MAE + prediction arrays for multi-clocks
train_one_epoch_multiclock   : Single training epoch, returns mean loss for multi-clocks
build_dataloaders   : Align, split, wrap methylation data into DataLoaders
train_one_epoch     : Single training epoch, returns mean loss
evaluate            : Run inference, return MAE + prediction arrays
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import DataLoader

from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

from epihydra import VocabWrapper, AgeDataset, MultiClockDataset

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data,
    vocab,
    *,
    clock_age    = 'age',
    train_ratio  = 0.70,
    valid_ratio  = 0.15,
    test_ratio   = 0.15,
    mask_ratio   = 0.15,
    batch_size   = 16,
    num_workers  = 0,
    random_state = 42,
    verbose      = True,
    scaler       = None,
):
    """
    Align methylation data to vocab probe order, split, and build DataLoaders.

    Training split receives random masking up to mask_ratio to encourage
    generalisation across incomplete CpG arrays. Validation and test splits
    are always unmasked so evaluation reflects full methylation profiles.

    If a pre-fitted scaler is provided, it is reused.
    Otherwise, a new MinMaxScaler is fitted on training ages.

    If train_ratio, valid_ratio, and test_ratio are all None, the entire
    dataset is returned as a single unmasked DataLoader under 'loader'.

    Parameters
    ----------
    data : dict
        'meta' : DataFrame with columns ['ID', 'age', ...]
        'CpG'  : DataFrame shaped (n_CpGs × n_samples)
                 Index = CpG probe IDs, Columns = sample IDs
    vocab        : VocabWrapper
    clock_age    : Name of the clock which is to be trained on e.g. chron, hannum, horvath
    train_ratio,
    valid_ratio,
    test_ratio   : float — must sum to 1.0, or all None to skip splitting
    mask_ratio   : float — upper bound on random masking applied to train split
    batch_size   : int
    num_workers  : int — 0 for Windows/macOS; ≥1 recommended on Linux cluster
    random_state : int
    verbose      : bool
    scaler       : MinMaxScaler fit on train ages or None

    Returns
    -------
    dict
        If splits provided:
            train_loader, valid_loader, test_loader : DataLoader
            train_df, valid_df, test_df             : DataFrame
            scaler                                  : MinMaxScaler fit on train ages
            n_present_cpgs, n_missing_cpgs          : int
        If no splits (all ratios None):
            loader                                  : DataLoader
            df                                      : DataFrame
            scaler                                  : MinMaxScaler fit on all ages
            n_present_cpgs, n_missing_cpgs          : int
    """

    split_mode = not (train_ratio is None and valid_ratio is None and test_ratio is None)

    if split_mode:
        assert None not in (train_ratio, valid_ratio, test_ratio), \
            "Either all ratios must be None, or all must be provided."
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + valid_ratio + test_ratio must sum to 1.0"

    # ── 1. Align meta and CpG on sample ID ──────────────────────────────────
    meta = data["meta"].set_index("ID").copy()
    cpg  = data["CpG"].copy()

    common_samples = meta.index.intersection(cpg.columns)
    meta = meta.loc[common_samples]
    cpg  = cpg[common_samples]

    if verbose:
        print(f"Samples in meta : {len(data['meta'])}")
        print(f"Samples in CpG  : {cpg.shape[1]}")
        print(f"Common samples  : {len(common_samples)}")

    # ── 2. Drop samples with NaN age ─────────────────────────────────────────
    valid_mask = meta[clock_age].notna()
    n_dropped  = (~valid_mask).sum()

    if n_dropped:
        if verbose:
            print(f"⚠  Dropping {n_dropped} sample(s) with NaN age.")
        meta = meta[valid_mask]
        cpg  = cpg[meta.index]

    if verbose:
        print(f"Samples after NaN drop : {len(meta)}")

    # ── 3. Reorder CpGs to match vocab probe order ───────────────────────────
    vocab_cpg_order = [
        vocab.vocab.lookup_token(int(i)) for i in vocab.CpG_ids
    ]

    data_cpgs    = set(cpg.index)
    present_cpgs = [c for c in vocab_cpg_order if c in data_cpgs]
    missing_cpgs = [c for c in vocab_cpg_order if c not in data_cpgs]

    if verbose:
        pct = 100 * len(present_cpgs) / len(vocab_cpg_order)
        print(f"\nVocab CpGs : {len(vocab_cpg_order):,} | "
              f"Present : {len(present_cpgs):,} ({pct:.1f}%) | "
              f"Missing : {len(missing_cpgs):,}")

        if missing_cpgs:
            print("ℹ  Missing CpGs → NaN → padded during tokenisation.")

    cpg_aligned = pd.DataFrame(
        np.nan,
        index   = meta.index,
        columns = vocab_cpg_order,
        dtype   = np.float32,
    )

    if present_cpgs:
        cpg_aligned[present_cpgs] = cpg.loc[present_cpgs].T.values

    # ── 4. Build combined DataFrame ──────────────────────────────────────────
    combined_df = pd.DataFrame(
        {clock_age: meta[clock_age].values, "data": list(cpg_aligned.values)},
        index=meta.index,
    )

    # ── 5. No-split mode: return a single DataLoader ─────────────────────────
    if not split_mode:
        if scaler is None:
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaler.fit(combined_df[clock_age].to_numpy().reshape(-1, 1))
            if verbose:
                print("ℹ  Scaler fitted on all ages (no-split mode).")
        else:
            if verbose:
                print("ℹ  Reusing supplied pre-fitted scaler.")

        dataset = AgeDataset(vocab, combined_df, scaler, clock_age, mask_ratio=0.0)
        loader  = DataLoader(
            dataset,
            batch_size  = batch_size,
            collate_fn  = dataset.collater,
            shuffle     = False,
            num_workers = num_workers,
        )

        if verbose:
            print(f"✓ Single loader built ({len(combined_df)} samples, no masking).\n")

        return {
            "loader"         : loader,
            "df"             : combined_df,
            "scaler"         : scaler,
            "n_present_cpgs" : len(present_cpgs),
            "n_missing_cpgs" : len(missing_cpgs),
        }

    # ── 6. Train / valid / test split ────────────────────────────────────────
    train_df, temp_df = train_test_split(
        combined_df,
        test_size    = round(1.0 - train_ratio, 10),
        random_state = random_state,
    )

    relative_valid = valid_ratio / (valid_ratio + test_ratio)

    valid_df, test_df = train_test_split(
        temp_df,
        test_size    = round(1.0 - relative_valid, 10),
        random_state = random_state,
    )

    if verbose:
        print(f"\nTrain : {len(train_df)} | Valid : {len(valid_df)} | Test : {len(test_df)}")
        print(f"Age range (train) : "
              f"{train_df[clock_age].min():.1f} – {train_df[clock_age].max():.1f} years")

    # ── 7. Fit or reuse scaler ───────────────────────────────────────────────
    if scaler is None:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_df[clock_age].to_numpy().reshape(-1, 1))

        if verbose:
            print("ℹ  Scaler fitted on training ages.")
    else:
        if verbose:
            print("ℹ  Reusing supplied pre-fitted scaler.")

    # ── 8. Build AgeDataset for each split ──────────────────────────────────
    train_dataset = AgeDataset(vocab, train_df, scaler, clock_age, mask_ratio=mask_ratio)
    valid_dataset = AgeDataset(vocab, valid_df, scaler, clock_age, mask_ratio=0.0)
    test_dataset  = AgeDataset(vocab, test_df,  scaler, clock_age, mask_ratio=0.0)

    # ── 9. Build DataLoaders ─────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        collate_fn  = train_dataset.collater,
        shuffle     = True,
        drop_last   = False,
        num_workers = num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = batch_size,
        collate_fn  = valid_dataset.collater,
        num_workers = num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        collate_fn  = test_dataset.collater,
        num_workers = num_workers,
    )

    if verbose:
        print("✓ Data loaded and ready.\n")

    return {
        "train_loader"   : train_loader,
        "valid_loader"   : valid_loader,
        "test_loader"    : test_loader,
        "train_df"       : train_df,
        "valid_df"       : valid_df,
        "test_df"        : test_df,
        "scaler"         : scaler,
        "n_present_cpgs" : len(present_cpgs),
        "n_missing_cpgs" : len(missing_cpgs),
    }

def build_multiclock_dataloaders(
    data,
    vocab,
    *,
    clock_names,
    train_ratio  = 0.70,
    valid_ratio  = 0.15,
    test_ratio   = 0.15,
    mask_ratio   = 0.15,
    batch_size   = 16,
    num_workers  = 0,
    random_state = 42,
    verbose      = True,
    scalers      = None,
):
    """
    Same alignment/split logic as build_dataloaders, but carries several
    clock-age target columns at once instead of a single 'age' column.

    A sample is kept as long as it has a non-NaN value for AT LEAST ONE of
    `clock_names` (so one clock's coverage doesn't force-drop samples that
    are useful for a different clock's head). Per-clock NaNs are preserved
    and surfaced via a validity mask so the training loop can exclude that
    clock's loss term for that sample.

    Parameters
    ----------
    data        : dict — 'meta' (DataFrame with 'ID' + one column per clock),
                  'CpG' (DataFrame shaped n_CpGs x n_samples)
    vocab       : VocabWrapper
    clock_names : list[str] — column names in meta, e.g.
                  ['horvath_age', 'hannum_age', 'phenoage']
    train_ratio, valid_ratio, test_ratio, mask_ratio,
    batch_size, num_workers, random_state, verbose : see build_dataloaders
    scalers     : dict[str, MinMaxScaler] or None — pre-fitted scalers to
                  reuse, keyed by clock name (e.g. a saved scalers.pkl).
                  Clocks missing from the dict, or all clocks if None, get
                  a new MinMaxScaler fitted on their non-NaN train values.

    Returns
    -------
    dict with train_loader, valid_loader, test_loader, train_df, valid_df,
    test_df, scalers (dict[str, MinMaxScaler] — the scalers actually used,
    supplied or newly fitted), n_present_cpgs,
    n_missing_cpgs, n_present_per_clock (dict[str, int])
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + valid_ratio + test_ratio must sum to 1.0"
    assert len(clock_names) >= 1, "clock_names must be non-empty"

    # ── 1. Align meta and CpG on sample ID ──────────────────────────────────
    meta = data["meta"].set_index("ID").copy()
    cpg  = data["CpG"].copy()

    common_samples = meta.index.intersection(cpg.columns)
    meta = meta.loc[common_samples]
    cpg  = cpg[common_samples]

    for name in clock_names:
        if name not in meta.columns:
            raise KeyError(f"clock '{name}' not found in meta columns: {list(meta.columns)}")

    if verbose:
        print(f"Samples in meta : {len(data['meta'])}")
        print(f"Samples in CpG  : {cpg.shape[1]}")
        print(f"Common samples  : {len(common_samples)}")

    # ── 2. Drop samples with NaN across ALL clocks ───────────────────────────
    any_valid = meta[clock_names].notna().any(axis=1)
    n_dropped = (~any_valid).sum()
    if n_dropped:
        if verbose:
            print(f"⚠  Dropping {n_dropped} sample(s) with no label for any requested clock.")
        meta = meta[any_valid]
        cpg  = cpg[meta.index]

    if verbose:
        print(f"Samples after drop : {len(meta)}")
        for name in clock_names:
            n = meta[name].notna().sum()
            print(f"  {name:<20}: {n} labeled")

    # ── 3. Reorder CpGs to match vocab probe order ───────────────────────────
    vocab_cpg_order = [
        vocab.vocab.lookup_token(int(i)) for i in vocab.CpG_ids
    ]
    data_cpgs    = set(cpg.index)
    present_cpgs = [c for c in vocab_cpg_order if c     in data_cpgs]
    missing_cpgs = [c for c in vocab_cpg_order if c not in data_cpgs]

    if verbose:
        pct = 100 * len(present_cpgs) / len(vocab_cpg_order)
        print(f"\nVocab CpGs : {len(vocab_cpg_order):,} | "
              f"Present : {len(present_cpgs):,} ({pct:.1f}%) | "
              f"Missing : {len(missing_cpgs):,}")

    cpg_aligned = pd.DataFrame(
        np.nan,
        index   = meta.index,
        columns = vocab_cpg_order,
        dtype   = np.float32,
    )
    if present_cpgs:
        cpg_aligned[present_cpgs] = cpg.loc[present_cpgs].T.values

    # ── 4. Build combined DataFrame ──────────────────────────────────────────
    combined_dict = {name: meta[name].values for name in clock_names}
    combined_dict["data"] = list(cpg_aligned.values)
    combined_df = pd.DataFrame(combined_dict, index=meta.index)

    # ── 5. Train / valid / test split ────────────────────────────────────────
    train_df, temp_df = train_test_split(
        combined_df,
        test_size    = round(1.0 - train_ratio, 10),
        random_state = random_state,
    )
    relative_valid = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size    = round(1.0 - relative_valid, 10),
        random_state = random_state,
    )

    if verbose:
        print(f"\nTrain : {len(train_df)} | Valid : {len(valid_df)} | Test : {len(test_df)}")

    # ── 6. Fit or reuse one scaler per clock ─────────────────────────────────
    # Supplied scalers are reused as-is; any clock without one gets a new
    # scaler fitted on that clock's non-NaN train values.
    supplied = dict(scalers) if scalers is not None else {}
    scalers  = {}
    for name in clock_names:
        if name in supplied:
            scalers[name] = supplied[name]
            if verbose:
                print(f"ℹ  {name:<20}: reusing supplied pre-fitted scaler.")
            continue

        vals = train_df[name].dropna().to_numpy().reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        if len(vals) >= 2:
            scaler.fit(vals)
        else:
            # Degenerate case: fewer than 2 labeled train samples for this
            # clock. Fit on [0, 1] so transform doesn't crash; this clock's
            # head will just have near-nothing to learn from — worth a warning.
            print(f"⚠  Clock '{name}' has <2 labeled samples in the train "
                  f"split — its head will not learn anything meaningful.")
            scaler.fit(np.array([[0.0], [1.0]]))
        scalers[name] = scaler
        if verbose:
            print(f"ℹ  {name:<20}: scaler fitted on training values.")

    # ── 7. Build MultiClockDataset for each split ────────────────────────────
    train_dataset = MultiClockDataset(vocab, train_df, scalers, clock_names, mask_ratio=mask_ratio)
    valid_dataset = MultiClockDataset(vocab, valid_df, scalers, clock_names, mask_ratio=0.0)
    test_dataset  = MultiClockDataset(vocab, test_df,  scalers, clock_names, mask_ratio=0.0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=train_dataset.collater,
        shuffle=True, drop_last=False, num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collater,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=test_dataset.collater,
        num_workers=num_workers,
    )

    if verbose:
        print("✓ Multi-clock data loaded and ready.\n")

    return {
        "train_loader"        : train_loader,
        "valid_loader"        : valid_loader,
        "test_loader"         : test_loader,
        "train_df"            : train_df,
        "valid_df"            : valid_df,
        "test_df"             : test_df,
        "scalers"             : scalers,
        "n_present_cpgs"      : len(present_cpgs),
        "n_missing_cpgs"      : len(missing_cpgs),
        "n_present_per_clock" : {n: int(meta[n].notna().sum()) for n in clock_names},
    }


def train_one_epoch_multiclock(model, loader, optimizer, scaler_amp, device, clock_names):
    """
    One epoch of joint multi-clock head training.

    A single backbone forward pass per batch (model.encode) feeds every
    clock head; per-clock MSE is masked to only the samples that actually
    have a label for that clock, then summed into one loss for one backward
    pass covering all heads simultaneously.

    Returns
    -------
    dict[str, float] — mean training loss per clock this epoch
    """
    model.train()
    model.backbone.eval()  # backbone frozen: keep its dropout/BN off regardless of .train()

    per_clock_losses = {name: [] for name in clock_names}

    for gene_ids, masked_vals, _, ages, valid, ages_norm in loader:
        gene_ids    = gene_ids.to(device)
        masked_vals = masked_vals.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(gene_ids, masked_vals, clock_names=clock_names)

            total_loss = 0.0
            batch_losses = {}
            for name in clock_names:
                v = valid[name].to(device)
                if v.sum() == 0:
                    continue  # no labeled samples for this clock in this batch
                target = ages_norm[name].to(device)[v]
                pred   = preds[name][v]
                loss   = nn.functional.mse_loss(pred, target)
                total_loss = total_loss + loss
                batch_losses[name] = loss.item()

        if isinstance(total_loss, torch.Tensor):
            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

        for name, l in batch_losses.items():
            per_clock_losses[name].append(l)

    return {
        name: float(np.mean(losses)) if losses else float("nan")
        for name, losses in per_clock_losses.items()
    }


@torch.no_grad()
def evaluate_multiclock(model, loader, scalers, device, clock_names):
    """
    Run inference for every clock head, return per-clock MAE + predictions.

    Returns
    -------
    dict[str, dict] — keyed by clock name, each value has
        {"mae": float, "preds": np.ndarray, "labels": np.ndarray}
    (preds/labels only include samples that actually had a label for that
    clock; clocks with zero labeled samples in this loader return NaN mae
    and empty arrays.)
    """
    model.eval()

    all_preds  = {name: [] for name in clock_names}
    all_labels = {name: [] for name in clock_names}

    for gene_ids, _, target_vals, ages, valid, _ in loader:
        gene_ids    = gene_ids.to(device)
        target_vals = target_vals.to(device)

        with torch.cuda.amp.autocast():
            preds = model(gene_ids, target_vals, clock_names=clock_names)

        for name in clock_names:
            v = valid[name].numpy()
            if v.sum() == 0:
                continue
            pred_norm = preds[name].float().cpu().numpy()[v]
            pred_age  = scalers[name].inverse_transform(
                pred_norm.reshape(-1, 1)
            ).flatten()
            all_preds[name].extend(pred_age)
            all_labels[name].extend(ages[name].numpy()[v])

    results = {}
    for name in clock_names:
        preds  = np.array(all_preds[name])
        labels = np.array(all_labels[name])
        mae    = float(np.mean(np.abs(preds - labels))) if len(preds) else float("nan")
        results[name] = {"mae": mae, "preds": preds, "labels": labels}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler_amp, device, clip_grad=1.0):
    """
    Run one full training epoch.

    Returns
    -------
    float — mean training loss for the epoch
    """
    model.train()
    losses = []

    for gene_ids, masked_vals, _, _, ages_norm in loader:
        gene_ids    = gene_ids.to(device)
        masked_vals = masked_vals.to(device)
        ages_norm   = ages_norm.to(device).squeeze()

        with torch.cuda.amp.autocast():
            pred = model(gene_ids, masked_vals)
            loss = criterion(pred, ages_norm)

        optimizer.zero_grad()
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, scaler, device):
    """
    Run inference and compute MAE.

    Uses unmasked target_vals so the backbone sees the full methylation
    profile at evaluation time, consistent with valid/test AgeDataset
    instances being initialised with mask_ratio=0.0.

    Returns
    -------
    mae    : float
    preds  : np.ndarray — predicted ages in years
    labels : np.ndarray — true ages in years
    """
    model.eval()
    preds, labels = [], []

    for gene_ids, _, target_vals, ages, _ in loader:
        gene_ids    = gene_ids.to(device)
        target_vals = target_vals.to(device)

        with torch.cuda.amp.autocast():
            pred = model(gene_ids, target_vals)

        pred_age = scaler.inverse_transform(
            pred.float().cpu().numpy().reshape(-1, 1)
        ).flatten()
        preds.extend(pred_age)
        labels.extend(ages.numpy())

    preds  = np.array(preds)
    labels = np.array(labels)
    return float(np.mean(np.abs(preds - labels))), preds, labels