"""
eval_utils.py
=============
Helper functions for evaluating EpiHydra clocks:
  - single clock      : 01_EvalSingleClock.ipynb
  - multi clock       : 02_EvalMultiClock.ipynb
  - missing features  : 03_MissingFeatures.ipynb

The EpiHydra model is a MethylGPT backbone (LoRA-adapted) with one attention
pooling head per clock. The base checkpoint (`best_model.pt`) carries the head
trained on chronological age ('age'). Other clocks (e.g. 'phenoage_age') are
separate heads saved as `best_head_<clock>.pt` in a multi-clock run folder.

Layout
------
  1. Paths, clocks & scalers
  2. Checkpoint loading
  3. Model construction
  4. Inference
  5. Metrics (one core function, everything else builds on it)
  6. Plotting
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from epihydra import AgePredictor, MultiClockPredictor, load_backbone

DEFAULT_CLOCK = "age"
CKPT_NAME     = "best_model.pt"
LORA_TARGETS  = ["Wqkv", "out_proj", "linear1", "linear2"]
PAD_VALUE     = -2.0   # missing CpG / padding value used in tokenisation
MASK_VALUE    = -1.0   # mask token value used in training

METRICS       = ("mae", "pearson_r", "r2")
_HIGH_IS_GOOD = {"mae": False, "pearson_r": True, "r2": True}
_METRIC_LABEL = {"mae": "MAE (years)", "pearson_r": "Pearson r", "r2": "R²"}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Paths, clocks & scalers
# ═════════════════════════════════════════════════════════════════════════════

def _ckpt_file(ckpt):
    """Accept either `best_model.pt` itself or the folder that contains it."""
    ckpt = Path(ckpt)
    return ckpt / CKPT_NAME if ckpt.is_dir() else ckpt


def find_backbone_checkpoint(model_path):
    """Return the single .pt file in the MethylGPT model folder."""
    candidates = sorted(Path(model_path).glob("*.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .pt file in {model_path}, found {len(candidates)}: "
            f"{[c.name for c in candidates]}"
        )
    return candidates[0]


def available_clocks(head_path):
    """Clock names with a trained head file (best_head_<clock>.pt) in `head_path`."""
    return sorted(p.stem[len("best_head_"):] for p in Path(head_path).glob("best_head_*.pt"))
 
 
def resolve_clocks(clocks, head_path):
    """
    Clocks to evaluate, always starting with 'age'.
    clocks=None means "'age' + every head found in head_path".
    """
    if clocks is None:
        clocks = available_clocks(head_path)
    return list(dict.fromkeys([DEFAULT_CLOCK, *clocks]))


def load_scalers(head_path, epihydra_path=None, clock=None):
    """
    Load label scalers saved by a training run (`head_path/scalers.pkl`).
 
    clock=None : return the whole {clock: scaler} dict (for multi-clock loaders)
    clock=str  : return that clock's scaler (for single-clock loaders)
 
    Returns None where nothing is saved, in which case build_dataloaders /
    build_multiclock_dataloaders fit a fresh scaler on the training split.
    """
    path = Path(head_path) / 'scalers.pkl'
    if not path.exists():
        print(f"ℹ  No scaler file at {path}; scalers will be fitted.")
        return None
 
    with open(path, "rb") as f:
        scalers = pickle.load(f)

    if epihydra_path is not None : 
        epihydra_path = Path(epihydra_path) / 'scaler.pkl'
        with open(epihydra_path, "rb") as f:
            age_scaler = pickle.load(f)
    
        scalers['age'] = age_scaler
 
    if clock is None:
        return scalers
    if clock not in scalers:
        print(f"ℹ  No saved scaler for '{clock}'; a new scaler will be fitted.")
    return scalers.get(clock)
    

# ═════════════════════════════════════════════════════════════════════════════
# 2. Checkpoint loading
# ═════════════════════════════════════════════════════════════════════════════

def _load_model_state(ckpt, device="cpu"):
    """Full AgePredictor state dict from best_model.pt (file or folder)."""
    raw = torch.load(_ckpt_file(ckpt), map_location=device)
    return raw.get("model_state", raw)


def load_head_weights(clock, ckpt, head_path, device="cpu"):
    """
    Return (head_state_dict, meta) for one clock.

    - 'age'        : the head.* weights inside best_model.pt (meta is None)
    - other clocks : head_path/best_head_<clock>.pt (meta is its training info,
                     used to check it was trained on the same LoRA backbone)
    """
    if clock == DEFAULT_CLOCK:
        state = _load_model_state(ckpt, device)
        head  = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        if not head:
            raise KeyError(f"No 'head.*' keys in {_ckpt_file(ckpt)}; "
                           f"cannot load the '{clock}' head.")
        return head, None

    head_file = Path(head_path) / f"best_head_{clock}.pt"
    if not head_file.exists():
        raise FileNotFoundError(f"No head checkpoint for clock '{clock}' at {head_file}")
    meta = torch.load(head_file, map_location=device)
    return meta["head_state"], meta


def load_clock_state(ckpt, head_path, clock=DEFAULT_CLOCK, device="cpu"):
    """
    Full AgePredictor state dict for `clock`: best_model.pt with its head.*
    weights swapped for best_head_<clock>.pt (unchanged for 'age').
    """
    state = _load_model_state(ckpt, device)
    if clock != DEFAULT_CLOCK:
        head, _ = load_head_weights(clock, ckpt, head_path, device)
        state.update({f"head.{k}": v for k, v in head.items()})
    return state


# ═════════════════════════════════════════════════════════════════════════════
# 3. Model construction
# ═════════════════════════════════════════════════════════════════════════════

def default_lora_config(r, alpha, dropout=0.1):
    """LoRA config used at training time. Must match to load checkpoints."""
    return LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS,
                      lora_dropout=dropout)

def lora_config_from_ckpt(ckpt, dropout=0.1):
    """LoRA config exactly as used in training, read from best_model.pt's run_config."""
    raw = torch.load(_ckpt_file(ckpt), map_location="cpu")
    rc  = raw.get("run_config", {}) or {}
    if "lora_r" not in rc or "lora_alpha" not in rc:
        raise KeyError(f"{_ckpt_file(ckpt)} has no lora_r / lora_alpha in run_config; "
                       f"pass lora_config explicitly.")
    print(f"LoRA from checkpoint: r={rc['lora_r']}, alpha={rc['lora_alpha']}")
    return default_lora_config(r=int(rc["lora_r"]), alpha=int(rc["lora_alpha"]), dropout=dropout)


def _build_backbone(model_path, vocab, pretrain_config, device, lora_config=None):
    """Load pretrained MethylGPT; wrap with LoRA if a config is given."""
    backbone = load_backbone(find_backbone_checkpoint(model_path), vocab, pretrain_config)
    if lora_config is not None:
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()
    return backbone.to(device)


def build_age_model(model_path, vocab, pretrain_config, device, *,
                    ckpt, head_path=None, clock=DEFAULT_CLOCK,
                    num_heads=4, dropout=0.1):
    """
    Trained single-clock AgePredictor for inference.

    LoRA r/alpha are read from `ckpt` (best_model.pt or its folder); the head for
    `clock` is swapped in from head_path when clock != 'age'. Returns the model in
    eval mode.
    """
    lora_config = lora_config_from_ckpt(ckpt)
    backbone    = _build_backbone(model_path, vocab, pretrain_config, device, lora_config)

    model = AgePredictor(
        backbone,
        embed_dim = pretrain_config["layer_size"],
        pad_id    = vocab.vocab["<pad>"],
        num_heads = num_heads,
        dropout   = dropout,
    ).to(device)

    model.load_state_dict(load_clock_state(ckpt, head_path, clock, device))
    model.eval()
    return model


def build_multiclock_model(model_path, vocab, pretrain_config, device, *,
                           ckpt, head_path, clock_names,
                           num_heads=4, dropout=0.1):
    """
    Trained MultiClockPredictor for inference.

    LoRA adapters from `ckpt` (best_model.pt or its folder; r/alpha read from its
    run_config) are merged into a frozen backbone, plus one trained head per clock
    from head_path (best_head_<clock>.pt). Returns the model in eval mode.
    """
    from epihydra.run_multiclock import load_lora_adapters

    ckpt        = _ckpt_file(ckpt)
    lora_config = lora_config_from_ckpt(ckpt)

    # ── Backbone: load, merge LoRA, freeze ──────────────────────────────────
    backbone = _build_backbone(model_path, vocab, pretrain_config, device)
    backbone, _ = load_lora_adapters(backbone, ckpt, device,
                                     lora_r=lora_config.r,
                                     lora_alpha=lora_config.lora_alpha,
                                     merge=True)
    backbone.requires_grad_(False)
    backbone.eval()

    model = MultiClockPredictor(
        backbone,
        embed_dim   = pretrain_config["layer_size"],
        pad_id      = vocab.vocab["<pad>"],
        clock_names = clock_names,
        num_heads   = num_heads,
        dropout     = dropout,
    ).to(device)

    # ── Heads ────────────────────────────────────────────────────────────────
    for name in clock_names:
        head_state, meta = load_head_weights(name, ckpt, head_path)
        model.heads[name].load_state_dict(head_state)

        trained_on = ((meta or {}).get("lora") or {}).get("ckpt")
        if trained_on and _ckpt_file(trained_on).resolve() != ckpt.resolve():
            print(f"  ⚠  head '{name}' was trained on backbone {trained_on}, not {ckpt}")
        src = ckpt.name if meta is None else f"best_head_{name}.pt (epoch {meta.get('epoch')})"
        print(f"  loaded head {name:<20} <- {src}")

    model.eval()
    return model

# ═════════════════════════════════════════════════════════════════════════════
# 4. Inference
# ═════════════════════════════════════════════════════════════════════════════

def ordered_loader(loader, batch_size=None):
    """Same dataset, shuffle=False, so batch order matches the split DataFrame."""
    ds = loader.dataset
    return DataLoader(ds, batch_size=batch_size or loader.batch_size,
                      shuffle=False, collate_fn=ds.collater, num_workers=0)


def _apply_masking(gene_ids, values, pad_id, frac=0.0, remove_ids=None):
    """
    Hide observed CpGs from the model by setting them to MASK_VALUE.

    remove_ids : token IDs blanked in EVERY sample (assay-level failure)
    frac       : random fraction of each sample's remaining observed CpGs

    Padding, the CLS token and already-missing CpGs are never touched.
    """
    if frac <= 0 and remove_ids is None:
        return values

    values   = values.clone()
    observed = gene_ids.ne(pad_id) & values.ne(PAD_VALUE)
    observed[:, 0] = False                                  # CLS token

    if remove_ids is not None:
        hit = observed & torch.isin(gene_ids, remove_ids)
        values[hit] = MASK_VALUE
        observed &= ~hit

    if frac > 0:
        for i in range(gene_ids.shape[0]):
            idx = observed[i].nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            n_mask = max(1, int(len(idx) * frac))
            values[i, idx[torch.randperm(len(idx), device=idx.device)[:n_mask]]] = MASK_VALUE

    return values


@torch.no_grad()
def predict_multiclock(model, loader, scalers, device, clock_names,
                        mask_frac=0.0, remove_ids=None, seed=None, impute_ref=None):
    """
    One backbone forward pass per batch, fanned out to every clock head.

    Parameters
    ----------
    loader     : unshuffled DataLoader (see ordered_loader)
    mask_frac  : extra fraction of observed CpGs to mask (missing-feature tests)
    seed       : torch seed for the extra masking, so fractions are comparable
    impute_ref : optional tensor from reference_by_token(). If given, every missing
                 CpG (masked -1 or originally missing -2) is replaced by its
                 reference beta before the forward pass, i.e. the same per-CpG
                 imputation the linear clocks receive. CpGs without a reference
                 value stay missing.

    Returns
    -------
    DataFrame indexed by sample ID with '<clock>_pred' / '<clock>_true' columns.
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    if remove_ids is not None:
        remove_ids = torch.as_tensor(remove_ids, dtype=torch.long, device=device)

    preds  = {n: [] for n in clock_names}
    truths = {n: [] for n in clock_names}

    batch_ids = []
    for gene_ids, _, target_vals, ages, _, _, ids in loader:
        batch_ids.extend(ids)
        gene_ids = gene_ids.to(device)
        values   = _apply_masking(gene_ids, target_vals.to(device), model.pad_id,
                                  frac=mask_frac, remove_ids=remove_ids)

        if impute_ref is not None:
            fill    = impute_ref[gene_ids]                       # reference beta per position
            missing = values.lt(0) & gene_ids.ne(model.pad_id)   # masked (-1) or missing (-2)
            missing[:, 0] = False                                # never touch <cls>
            values  = torch.where(missing & ~torch.isnan(fill), fill, values)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(gene_ids, values, clock_names=clock_names)

        for name in clock_names:
            p = out[name].float().cpu().numpy().reshape(-1, 1)
            preds[name].append(scalers[name].inverse_transform(p).ravel())
            truths[name].append(ages[name].numpy())

    columns = {}
    for name in clock_names:
        columns[f"{name}_pred"] = np.concatenate(preds[name])
        columns[f"{name}_true"] = np.concatenate(truths[name])
    return pd.DataFrame(columns, index=pd.Index(batch_ids, name="ID"))


# ═════════════════════════════════════════════════════════════════════════════
# 5. Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(true, pred, min_samples=2):
    """
    The one place metrics are computed. NaN pairs are dropped first.

    R² is r2_score(true, pred). r and R² are NaN if either side is constant;
    everything except n is NaN if fewer than `min_samples` pairs remain.

    Returns
    -------
    dict with keys n, mae, pearson_r, r2
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok   = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[ok], pred[ok]

    out = {"n": int(len(true)), "mae": np.nan, "pearson_r": np.nan, "r2": np.nan}
    if out["n"] < min_samples:
        return out

    out["mae"] = float(np.mean(np.abs(pred - true)))
    if np.std(true) > 0 and np.std(pred) > 0:
        out["pearson_r"] = float(pearsonr(true, pred)[0])
        out["r2"]        = float(r2_score(true, pred))
    return out


def print_metrics(metrics, title=None):
    """Pretty-print a compute_metrics dict."""
    print(f"\n{'─' * 40}")
    if title:
        print(f"   {title}")
    print(f"   n        : {metrics['n']}")
    print(f"   MAE      : {metrics['mae']:.2f} years")
    print(f"   R²       : {metrics['r2']:.3f}")
    print(f"   Pearson r: {metrics['pearson_r']:.3f}")
    print(f"{'─' * 40}\n")


def metrics_by_clock(predictions, clock_names, min_samples=2, **extra):
    """
    Long table, one row per clock: each clock scored against its own labels.
    `extra` key/values are added as constant columns (e.g. mask_frac=0.2).
    """
    return pd.DataFrame([
        {"clock": c, **extra,
         **compute_metrics(predictions[f"{c}_true"], predictions[f"{c}_pred"], min_samples)}
        for c in clock_names
    ])


def pairwise_metrics(predictions, clock_names, rows="pred", cols="true", min_samples=2):
    """
    n×n tables comparing every clock against every other clock.

    rows='pred', cols='true' : cross-clock performance (diagonal = own labels)
    rows='true', cols='true' : agreement between the ground-truth clocks

    Columns are treated as the reference, so R² = r2_score(col, row).

    Returns
    -------
    dict {metric: DataFrame} for metric in ('mae', 'pearson_r', 'r2', 'n')
    """
    names = list(clock_names)
    index   = pd.Index(names, name=f"{rows.capitalize()} clock")
    columns = pd.Index(names, name=f"{cols.capitalize()} clock")
    out = {m: pd.DataFrame(np.nan, index=index, columns=columns) for m in (*METRICS, "n")}

    for r in names:
        for c in names:
            m = compute_metrics(predictions[f"{c}_{cols}"], predictions[f"{r}_{rows}"],
                                min_samples)
            for key, val in m.items():
                out[key].loc[r, c] = val
    return out


def evaluate_missing_fraction(model, loader, scalers, device, clock_names,
                              mask_fractions, seed=42, min_samples=2,
                              impute_ref=None):
    """
    Per-clock performance as an increasing fraction of observed CpGs is masked.
    The same seed is reused at every fraction so results are comparable.

    impute_ref : optional tensor from reference_by_token(). If given, every
                 missing CpG (masked or originally missing) is replaced by its
                 per-CpG reference beta before prediction -- the same imputation
                 the linear clocks receive. Use the same seed with and without
                 it so both runs hide the same CpGs.

    Returns
    -------
    Long DataFrame [clock, mask_frac, imputed, n, mae, pearson_r, r2]
    """
    frames = [
        metrics_by_clock(
            predict_multiclock(model, loader, scalers, device, clock_names,
                               mask_frac=frac, seed=seed, impute_ref=impute_ref),
            clock_names, min_samples,
            mask_frac=frac, imputed=impute_ref is not None,
        )
        for frac in mask_fractions
    ]
    return pd.concat(frames, ignore_index=True)


def evaluate_systematic_removal(model, loader, scalers, device, clock_names,
                                ranked_indices, removal_fractions, min_samples=2,
                                impute_ref=None):
    """
    Remove the same CpG probes from EVERY sample, simulating an assay or
    genomic-region failure, in the order given by `ranked_indices`.

    Parameters
    ----------
    ranked_indices    : positions in the vocab-aligned CpG array (the 'data'
                        column), in removal order; index 0 is removed first.
                        From rank_cpgs().
    removal_fractions : fractions of len(ranked_indices) to remove,
                        e.g. [0.0, 0.1, ..., 0.9]
    impute_ref        : optional tensor from reference_by_token(). If given,
                        removed and originally missing CpGs are replaced by
                        their per-CpG reference beta before prediction.

    Returns
    -------
    Long DataFrame [clock, removal_frac, n_removed, total_missing, imputed,
                    n, mae, pearson_r, r2]
    total_missing is the mean fraction of vocab CpGs whose measured value the
    model did not get (real missingness + removed probes), comparable with
    evaluate_missing_fraction. With impute_ref those CpGs carry reference
    values instead, but they are still counted as missing measurements.
    """
    ranked_indices = np.asarray(ranked_indices)
    cpg_ids        = loader.dataset.vocab.CpG_ids             # array position → token ID
    missing        = np.stack([~np.isfinite(x) for x in loader.dataset.gene_datas])
    n_vocab        = missing.shape[1]
    base_missing   = missing.sum(axis=1)

    frames = []
    for frac in removal_fractions:
        n_remove = int(frac * len(ranked_indices))
        removed  = ranked_indices[:n_remove]

        # Removed probes that were already missing don't add to the total
        hidden = base_missing + n_remove - missing[:, removed].sum(axis=1)

        preds = predict_multiclock(
            model, loader, scalers, device, clock_names,
            remove_ids=cpg_ids[removed] if n_remove else None,
            impute_ref=impute_ref,
        )
        frames.append(metrics_by_clock(
            preds, clock_names, min_samples,
            removal_frac=frac, n_removed=n_remove,
            total_missing=float((hidden / n_vocab).mean()),
            imputed=impute_ref is not None,
        ))
    return pd.concat(frames, ignore_index=True)


def reference_by_token(reference, vocab, device):
    """
    Tensor indexed by token ID holding each CpG's reference beta
    (NaN where a CpG has no reference value).

    reference : pd.Series indexed by CpG name, e.g. from build_cpg_reference
    vocab     : VocabWrapper
    """
    ref    = torch.full((len(vocab.vocab),), float("nan"), dtype=torch.float32)
    known  = reference.dropna()
    names  = [c for c in known.index if c in vocab.vocab]
    if names:
        ids = torch.as_tensor(vocab.vocab.lookup_indices(names), dtype=torch.long)
        ref[ids] = torch.as_tensor(known.loc[names].to_numpy(), dtype=torch.float32)
    print(f"Reference values for {len(names):,} of {len(vocab.CpG_list):,} vocab CpGs")
    return ref.to(device)
    
    
def compute_sample_coverage(df):
    """
    Fraction of vocab CpGs actually observed for each sample.

    df : any DataFrame from build_*dataloaders with a 'data' column,
         e.g. splits['test_df'] or dataloader['df'].
    """
    coverage = [np.isfinite(row).mean() for row in df["data"]]
    return pd.Series(coverage, index=df.index, name="cpg_coverage")


def coverage_performance(predictions, coverage, clock_names, n_bins=5, min_samples=3):
    """
    Per-clock metrics within equal-width bins of real CpG coverage.

    Returns
    -------
    Long DataFrame [clock, coverage_bin, coverage_mid, n, mae, pearson_r, r2];
    bins with fewer than `min_samples` labelled samples are dropped.
    """
    merged = predictions.join(coverage.rename("cpg_coverage"), how="inner")
    edges  = np.linspace(0, 1, n_bins + 1)
    mids   = (edges[:-1] + edges[1:]) / 2
    labels = [f"{edges[i]:.0%}–{edges[i + 1]:.0%}" for i in range(n_bins)]
    codes  = pd.cut(merged["cpg_coverage"], bins=edges, labels=False, include_lowest=True)

    frames = [
        metrics_by_clock(grp, clock_names, min_samples,
                         coverage_bin=labels[int(code)], coverage_mid=mids[int(code)])
        for code, grp in merged.groupby(codes)
    ]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result[result["n"] >= min_samples].reset_index(drop=True)


def rank_cpgs(df, strategy="low_variance"):
    """
    Return a vocab-ordered array of CpG token indices, ranked by a chosen
    strategy so we know which probes to remove first in systematic removal.
 
    Parameters
    ----------
    strategy : "low_variance"  — remove lowest-variance probes first
                                 (least informative across the cohort)
               "high_variance" — remove highest-variance probes first
                                 (most informative across the cohort)
               "mean"          — remove probes with highest absolute mean
                                 methylation first (most extreme first)
 
    Returns
    -------
    ranked_indices : np.ndarray of int — vocab positions, ordered so that
                     index 0 is removed first
    strategy_label : str for plot titles / legends
    """
    strategy_labels = {
        "low_variance"  : "ascending variance (least variable first)",
        "high_variance" : "descending variance (most variable first)",
        "mean"          : "descending mean absolute value (most extreme first)",
    }
    if strategy not in strategy_labels:
        raise ValueError(f"strategy must be one of {list(strategy_labels)}; got '{strategy}'")

    X  = np.stack(df["data"]) # (n_samples, n_cpgs), may contain NaN

    with np.errstate(all="ignore"):
        if strategy in ("low_variance", "high_variance"):
            stat = np.nanvar(X, axis=0)
        else:  # mean
            stat = np.nanmean(np.abs(X), axis=0)

    if strategy == "low_variance":
        # NaN probes go last — already missing so removing them has no effect
        stat  = np.where(np.isfinite(stat), stat, np.inf)
        ranked = np.argsort(stat)           # ascending: lowest variance first
    elif strategy == "high_variance":
        stat  = np.where(np.isfinite(stat), stat, -np.inf)
        ranked = np.argsort(stat)[::-1]    # descending: highest variance first
    else:  # mean
        stat  = np.where(np.isfinite(stat), stat, -np.inf)
        ranked = np.argsort(stat)[::-1]    # descending: highest mean first

    return ranked.astype(int), strategy_labels[strategy]

# ═════════════════════════════════════════════════════════════════════════════
# 6. Plotting
# ═════════════════════════════════════════════════════════════════════════════

# Typographic / style constants matching Nature Medicine figure conventions
_NM_RC = {
    "font.family"        : "sans-serif",
    "font.sans-serif"    : ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size"          : 7,
    "axes.titlesize"     : 8,
    "axes.labelsize"     : 7,
    "xtick.labelsize"    : 6,
    "ytick.labelsize"    : 6,
    "legend.fontsize"    : 6,
    "axes.linewidth"     : 0.75,
    "xtick.major.width"  : 0.75,
    "ytick.major.width"  : 0.75,
    "xtick.major.size"   : 3,
    "ytick.major.size"   : 3,
    "lines.linewidth"    : 1.0,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "figure.dpi"         : 300,
    "savefig.dpi"        : 300,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.02,
}
_MM = 1 / 25.4                    # mm → inches (NM single column = 89 mm)

_SCATTER_FACE = "#2166AC"
_SCATTER_EDGE = "#1a1a1a"
_IDENTITY_COL = "#D6604D"
_REG_COL      = "#4D4D4D"


def _clean(name):
    """Readable label from a clock column name."""
    return name.replace("_age", "").replace("_clock", "").replace("_", " ").title()


def _cmap(metric):
    """Green = good for every metric."""
    return "RdYlGn" if _HIGH_IS_GOOD[metric] else "RdYlGn_r"


def _save(fig, save_path, formats=("svg",)):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(save_path.with_suffix(f".{fmt}"))
        print(f"Saved → {save_path.with_suffix(f'.{fmt}')}")


def _scatter_panel(ax, true, pred, xlabel, ylabel, title,
                   identity=True, regression=True, annotate=True):
    """
    One predicted-vs-reference scatter: points, y = x, regression line and
    a stats annotation. Returns the compute_metrics dict.
    """
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    stats = compute_metrics(true, pred)

    ax.scatter(true, pred, s=8, facecolor=_SCATTER_FACE, edgecolor=_SCATTER_EDGE,
               linewidths=0.3, alpha=0.75, zorder=3, rasterized=True)

    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    pad    = (hi - lo) * 0.04
    lims   = np.array([lo - pad, hi + pad])

    if identity:
        ax.plot(lims, lims, color=_IDENTITY_COL, lw=0.9, ls="--",
                alpha=0.8, zorder=2, label="y = x")
    if regression and stats["n"] >= 2 and np.std(true) > 0:
        m, b = np.polyfit(true, pred, 1)
        ax.plot(lims, m * lims + b, color=_REG_COL, lw=1.0,
                alpha=0.9, zorder=2, label=f"slope = {m:.2f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    if identity or regression:
        ax.legend(frameon=False, handlelength=1.5, handletextpad=0.4, loc="lower right")

    if annotate:
        ax.text(0.05, 0.95,
                f"r = {stats['pearson_r']:.2f}\nR² = {stats['r2']:.2f}\n"
                f"MAE = {stats['mae']:.1f}\nn = {stats['n']}",
                transform=ax.transAxes, fontsize=6, va="top",
                linespacing=1.6, color="#1a1a1a")
    return stats


# ── Scatter plots ────────────────────────────────────────────────────────────

def plot_predictions(labels, preds, clock=DEFAULT_CLOCK, ax=None,
                     save_path=None, fig_width_mm=89):
    """Single-clock predicted vs. target scatter (notebook 01)."""
    target = "chronological age" if clock == DEFAULT_CLOCK else _clean(clock)
    with plt.rc_context(_NM_RC):
        if ax is None:
            w = fig_width_mm * _MM
            _, ax = plt.subplots(figsize=(w, w), constrained_layout=True)
        stats = _scatter_panel(ax, labels, preds,
                               xlabel=f"True {target} (years)",
                               ylabel="Predicted age (years)",
                               title=_clean(clock))
    _save(ax.figure, save_path)
    return ax, stats


def plot_clock_panels(predictions, clock, chron_clock=DEFAULT_CLOCK,
                      save_path=None, fig_width_mm=89):
    """
    Two-panel figure for one clock:
      A — predicted <clock> vs. true <clock>
      B — predicted <clock> vs. chronological age

    Returns
    -------
    fig, (ax_a, ax_b), {"pred_vs_true": stats, "pred_vs_chron": stats}
    """
    cols = [f"{clock}_pred", f"{clock}_true", f"{chron_clock}_true"]
    df   = predictions[cols].dropna()
    if len(df) < 2:
        raise ValueError(f"Fewer than 2 samples with all of {cols} present.")

    label = _clean(clock)
    w     = fig_width_mm * _MM

    with plt.rc_context(_NM_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(w * 2, w * 0.52),
                                         constrained_layout=True)
        stats_a = _scatter_panel(
            ax_a, df[f"{clock}_true"], df[f"{clock}_pred"],
            xlabel=f"True {label} (years)", ylabel=f"Predicted {label} (years)",
            title=f"A   Predicted vs. true {label}",
        )
        stats_b = _scatter_panel(
            ax_b, df[f"{chron_clock}_true"], df[f"{clock}_pred"],
            xlabel="Chronological age (years)", ylabel=f"Predicted {label} (years)",
            title=f"B   Predicted {label} vs. chronological age",
        )

    _save(fig, save_path)
    return fig, (ax_a, ax_b), {"pred_vs_true": stats_a, "pred_vs_chron": stats_b}


def plot_all_clock_panels(predictions, clock_names, chron_clock=DEFAULT_CLOCK,
                          save_dir=None, fig_width_mm=89):
    """plot_clock_panels for every clock except chron_clock. Returns {clock: result}."""
    results = {}
    for clock in (c for c in clock_names if c != chron_clock):
        try:
            results[clock] = plot_clock_panels(
                predictions, clock, chron_clock=chron_clock,
                save_path=Path(save_dir) / clock if save_dir else None,
                fig_width_mm=fig_width_mm,
            )
            plt.show()
        except ValueError as e:
            print(f"⚠  Skipping '{clock}': {e}")
    return results


# ── Tables & heatmaps ────────────────────────────────────────────────────────

def style_metric_table(table, metric):
    """Green-to-red pandas Styler for any n×n metric table."""
    return (table.style
                 .background_gradient(cmap=_cmap(metric), axis=None)
                 .format("{:.3f}", na_rep="—")
                 .set_caption(metric))


def plot_metric_heatmap(table, metric, title=None, mask_diagonal=True, save_path=None):
    """
    Heatmap of one n×n table from pairwise_metrics. Colour limits use the
    visible (off-diagonal, non-NaN) cells only.
    """
    table = table.astype(float).copy()
    table.index   = pd.Index([_clean(n) for n in table.index],   name=table.index.name)
    table.columns = pd.Index([_clean(n) for n in table.columns], name=table.columns.name)

    k    = len(table)
    diag = np.eye(k, dtype=bool) if mask_diagonal else np.zeros((k, k), dtype=bool)
    mask = diag | table.isna().to_numpy()
    vis  = table.to_numpy()[~mask]
    vmin, vmax = (vis.min(), vis.max()) if vis.size else (None, None)

    fig, ax = plt.subplots(figsize=(k + 3, k + 3))
    sns.heatmap(table, annot=True, fmt=".2f", cmap=_cmap(metric), mask=mask,
                linewidths=0.5, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title(title or _METRIC_LABEL[metric], pad=12)
    ax.set_xlabel(table.columns.name)
    ax.set_ylabel(table.index.name)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Missing-feature plots ────────────────────────────────────────────────────

def plot_missing_fraction_curves(mask_results, metric="mae", clock_names=None,
                                 x="total_missing", ax=None, save_path=None):
    """
    Line per clock: performance vs. missing CpGs (evaluate_missing_fraction).

    x : "total_missing" — real + injected missingness (% of all vocab CpGs)
        "mask_frac"     — injected masking only (% of observed CpGs)
    """
    clock_names = clock_names or mask_results["clock"].unique().tolist()
    xlabel = {"total_missing": "Total CpGs missing (%)",
              "mask_frac":     "Additional CpGs masked (%)"}[x]

    with plt.rc_context(_NM_RC):
        if ax is None:
            fig, ax = plt.subplots(figsize=(89 * _MM, 70 * _MM), constrained_layout=True)
        else:
            fig = ax.figure

        cmap = plt.get_cmap("tab10")
        for i, name in enumerate(clock_names):
            sub = mask_results[mask_results["clock"] == name].sort_values(x)
            ax.plot(sub[x] * 100, sub[metric], marker="o", markersize=3,
                    lw=1.0, color=cmap(i), label=_clean(name))

        if x == "total_missing":
            base = mask_results["total_missing"].min()
            ax.axvline(base * 100, color="grey", lw=0.6, ls=":", zorder=0)
            ax.set_xlim(right=100)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(_METRIC_LABEL[metric])
        ax.set_title("Clock performance vs. missing CpGs", pad=4, fontweight="bold")
        ax.legend(frameon=False, loc="best")
        if not _HIGH_IS_GOOD[metric]:
            ax.set_ylim(bottom=0)

    _save(fig, save_path)
    return fig, ax


def plot_systematic_removal_curves(systematic_results, random_results=None,
                                   metric="mae", clock_names=None,
                                   strategy_label="systematic",
                                   x="total_missing", save_path=None):
    """
    One subplot per clock: performance vs. fraction of probes systematically
    removed. If `random_results` (evaluate_missing_fraction) is given, the random
    masking curve is overlaid dashed, so the gap shows whether probe identity
    matters or any missing data hurts equally.
    """
    clock_names = clock_names or systematic_results["clock"].unique().tolist()
    n_clocks    = len(clock_names)

    with plt.rc_context(_NM_RC):
        fig, axes = plt.subplots(1, n_clocks, figsize=(min(183, 55 * n_clocks) * _MM, 65 * _MM),
                                 sharey=True, constrained_layout=True, squeeze=False)
        axes = axes[0]
        cmap = plt.get_cmap("tab10")

        for i, (ax, name) in enumerate(zip(axes, clock_names)):
            sub = systematic_results[systematic_results["clock"] == name].sort_values(x)
            ax.plot(sub[x] * 100, sub[metric], marker="o", markersize=3,
                    lw=1.0, color=cmap(i), label=strategy_label)

            if random_results is not None:
                rsub = random_results[random_results["clock"] == name].sort_values(x)
                ax.plot(rsub[x] * 100, rsub[metric], marker="s", markersize=3,
                        lw=1.0, color=cmap(i), ls="--", alpha=0.6, label="random")

            ax.set_title(_clean(name), pad=4, fontweight="bold")
            ax.set_xlabel("Total CpGs missing (%)" if x == "total_missing" else "CpGs removed (%)")
            if not _HIGH_IS_GOOD[metric]:
                ax.set_ylim(bottom=0)
            ax.legend(frameon=False, fontsize=5)

        axes[0].set_ylabel(_METRIC_LABEL[metric])
        fig.suptitle(f"Systematic probe removal — {_METRIC_LABEL[metric]}",
                     fontsize=8, fontweight="bold")

    _save(fig, save_path)
    return fig, axes


def plot_coverage_performance(coverage_results, metric="mae", clock_names=None,
                              save_path=None):
    """One subplot per clock, one bar per real-coverage bin (coverage_performance)."""
    clock_names = clock_names or coverage_results["clock"].unique().tolist()
    n_clocks    = len(clock_names)

    with plt.rc_context(_NM_RC):
        fig, axes = plt.subplots(1, n_clocks, figsize=(min(183, 50 * n_clocks) * _MM, 60 * _MM),
                                 sharey=True, constrained_layout=True, squeeze=False)
        axes = axes[0]
        cmap = plt.get_cmap("Blues")

        for ax, name in zip(axes, clock_names):
            ax.set_title(_clean(name), pad=4, fontweight="bold")
            sub = coverage_results[coverage_results["clock"] == name].sort_values("coverage_mid")
            if sub.empty:
                continue

            colours = [cmap(0.35 + 0.6 * i / max(len(sub) - 1, 1)) for i in range(len(sub))]
            bars = ax.bar(sub["coverage_bin"], sub[metric], color=colours,
                          edgecolor="white", linewidth=0.4, width=0.6)
            offset = np.nanmax(np.abs(sub[metric])) * 0.02
            for bar, n in zip(bars, sub["n"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                        f"n={n}", ha="center", va="bottom", fontsize=5)

            ax.set_xlabel("CpG coverage")
            ax.tick_params(axis="x", rotation=30)
            if not _HIGH_IS_GOOD[metric]:
                ax.set_ylim(bottom=0)

        axes[0].set_ylabel(_METRIC_LABEL[metric])
        fig.suptitle(f"{_METRIC_LABEL[metric]} by real CpG coverage",
                     fontsize=8, fontweight="bold")

    _save(fig, save_path)
    return fig, axes