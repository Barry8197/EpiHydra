"""
eval_utils.py
=============
Helper functions for evaluating EpiHydra clocks:
  - single clock : 01_EvalSingleClock.ipynb
  - multi clock  : 02_EvalMultiClock.ipynb

The EpiHydra model is a MethylGPT backbone (LoRA-adapted) with an AgePredictor
head. The base checkpoint (`best_model.pt`) carries the head trained on
chronological age ('age'). Other clocks (e.g. 'phenoage_age') are separate
heads saved as `best_head_<clock>.pt`, which are swapped into the base
checkpoint before loading.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from epihydra import AgePredictor, load_backbone  # noqa: E402

# Name of the default clock, whose head lives inside best_model.pt itself
DEFAULT_CLOCK = "age"

# ─────────────────────────────────────────────────────────────────────────────
# Scalers
# ─────────────────────────────────────────────────────────────────────────────
def load_scaler(clock, head_path, scaler_file="scalers.pkl"):
    """
    Return the pre-fitted label scaler for `clock`, or None if there isn't one.

    Scalers for the clock heads are stored together in `head_path/scalers.pkl`
    as a dict {clock_name: scaler}. If the clock is not in that dict (typically
    the default 'age' clock), None is returned and `build_dataloaders` will fit
    a new scaler on the training split instead.
    """
    scaler_path = Path(head_path) / scaler_file
    if not scaler_path.exists():
        print(f"ℹ  No scaler file at {scaler_path}; a new scaler will be fitted.")
        return None

    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    if clock not in scalers:
        print(f"ℹ  No saved scaler for '{clock}'; a new scaler will be fitted.")
        return None
    return scalers[clock]


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────
def find_backbone_checkpoint(model_path):
    """Return the single .pt file in the MethylGPT model folder."""
    candidates = sorted(Path(model_path).glob("*.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .pt file in {model_path}, found {len(candidates)}: "
            f"{[c.name for c in candidates]}"
        )
    return candidates[0]

def _build_backbone(model_path, vocab, pretrain_config, device, lora_config=None):
    """
    Shared backbone builder: load pretrained weights and optionally wrap with LoRA.

    Parameters
    ----------
    lora_config : LoraConfig or None — if None, backbone is returned as-is (frozen,
                  no LoRA). If provided, backbone is wrapped with get_peft_model
                  ready for training.
    """
    backbone = load_backbone(find_backbone_checkpoint(model_path), vocab, pretrain_config)

    if lora_config is not None:
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    return backbone.to(device)


def build_age_model(model_path, vocab, pretrain_config, device,
                    num_heads=4, dropout=0.1, lora_config=None):
    """
    Build an AgePredictor ready for training.

    Backbone is wrapped with LoRA (default r=16, alpha=8) unless a custom
    lora_config is supplied. No weights are loaded into the head.

    Parameters
    ----------
    model_path      : str | Path — directory containing the backbone checkpoint
    vocab           : VocabWrapper
    pretrain_config : dict
    device          : torch.device
    num_heads       : int
    dropout         : float
    lora_config     : LoraConfig or None — defaults to r=16, alpha=8
    """
    if lora_config is None:
        lora_config = LoraConfig(
            r             = 16,
            lora_alpha    = 8,
            target_modules= ["Wqkv", "out_proj", "linear1", "linear2"],
            lora_dropout  = 0.1,
        )

    backbone = _build_backbone(model_path, vocab, pretrain_config, device, lora_config)

    model = AgePredictor(
        backbone,
        embed_dim = pretrain_config["layer_size"],
        pad_id    = vocab.vocab["<pad>"],
        num_heads = num_heads,
        dropout   = dropout,
    ).to(device)

    return model


def build_multiclock_model(model_path, vocab, pretrain_config, lora_ckpt, head_path,
                           clock_names, device, num_heads=4, dropout=0.1,
                           lora_r=16, lora_alpha=8):
    """
    Build a MultiClockPredictor with a frozen, LoRA-merged backbone and one
    trained head per clock.

    Unlike build_age_model, this loads a trained checkpoint — backbone weights
    are merged and frozen, and each clock head is loaded from disk.

    Parameters
    ----------
    model_path      : str | Path — directory containing the backbone checkpoint
    vocab           : VocabWrapper
    pretrain_config : dict
    lora_ckpt       : str | Path — directory containing LoRA adapter weights
                      and head checkpoints (best_head_<name>.pt)
    head_path       : str | Path — fallback path if heads not in lora_ckpt
    clock_names     : list[str]
    device          : torch.device
    num_heads       : int
    dropout         : float
    lora_r          : int   — must match training config
    lora_alpha      : int   — must match training config
    """
    from epihydra import MultiClockPredictor
    from run_multiclock import load_lora_adapters

    lora_ckpt = Path(lora_ckpt)

    # ── Backbone: load, merge LoRA, freeze ──────────────────────────────────
    backbone = _build_backbone(model_path, vocab, pretrain_config, device)
    backbone, lora_info = load_lora_adapters(
        backbone, lora_ckpt, device, lora_r=lora_r, lora_alpha=lora_alpha, merge=True,
    )
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    # ── Model ────────────────────────────────────────────────────────────────
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
        head_state, meta = load_head_weights(name, lora_ckpt, head_path)
        model.heads[name].load_state_dict(head_state)

        trained_on = ((meta or {}).get("lora") or {}).get("ckpt")
        if trained_on and Path(trained_on).resolve() != lora_ckpt.resolve():
            print(f"  ⚠  head '{name}' was trained on backbone {trained_on}, "
                  f"not {lora_ckpt}")
        src = lora_ckpt.name if meta is None else f"best_head_{name}.pt (epoch {meta.get('epoch')})"
        print(f"  loaded head {name:<20} <- {src}")

    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
def load_clock_state(ckpt_path, head_path, clock=DEFAULT_CLOCK, device="cpu"):
    """
    Return a model state dict for the requested clock.

    - clock == 'age' : the state dict from `ckpt_path/best_model.pt` as-is
                       (its head is already the chronological-age head).
    - any other clock: the same state dict, with the `head.*` weights replaced
                       by those in `head_path/best_head_<clock>.pt`.
    """
    ckpt = torch.load(Path(ckpt_path) / "best_model.pt", map_location=device)
    state = ckpt.get("model_state", ckpt)

    if clock == DEFAULT_CLOCK:
        return state

    head_file = Path(head_path) / f"best_head_{clock}.pt"
    if not head_file.exists():
        raise FileNotFoundError(f"No head checkpoint for clock '{clock}' at {head_file}")

    head_state = torch.load(head_file, map_location=device)["head_state"]
    for k, v in head_state.items():
        state[f"head.{k}"] = v
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & plotting
# ─────────────────────────────────────────────────────────────────────────────
def eval_metrics(labels, preds, mae, verbose=True):
    """
    Compute R² and Pearson r for a set of predictions and (optionally) print
    them alongside the MAE.

    Returns
    -------
    dict with keys 'mae', 'r2', 'pearson_r'.
    """
    labels, preds = np.asarray(labels), np.asarray(preds)
    metrics = {
        "mae": float(mae),
        "r2": float(r2_score(labels, preds)),
        "pearson_r": float(pearsonr(labels, preds)[0]),
    }

    if verbose:
        print(f"\n{'─' * 40}")
        print(f"   MAE      : {metrics['mae']:.2f} years")
        print(f"   R²       : {metrics['r2']:.3f}")
        print(f"   Pearson r: {metrics['pearson_r']:.3f}")
        print(f"{'─' * 40}\n")
    return metrics


def plot_predictions(labels, preds, metrics, clock=DEFAULT_CLOCK, ax=None):
    """Scatter of predicted vs. target age with a y = x reference line."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Black underlay + translucent blue overlay gives each point an outline
    ax.scatter(labels, preds, s=15, c="black", alpha=1.0, zorder=1)
    ax.scatter(labels, preds, s=10, c="steelblue", alpha=0.5, zorder=2)

    lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
    ax.plot(lims, lims, "r-", alpha=0.8, label="y = x")

    target = "Chronological age" if clock == DEFAULT_CLOCK else f"Target {clock}"
    ax.set_xlabel(f"{target} (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title(f"{clock} | MAE={metrics['mae']:.1f} yrs | "
                 f"r={metrics['pearson_r']:.3f}")
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    return ax


# ═════════════════════════════════════════════════════════════════════════════
# Multi-clock evaluation (02_EvalMultiClock.ipynb)
# ═════════════════════════════════════════════════════════════════════════════
#
# The clock heads were trained by run_multiclock.py on a FROZEN backbone whose
# LoRA adapters (from best_model.pt) were merged into the base weights. We
# rebuild exactly that backbone, then load every head into a single
# MultiClockPredictor:
#   - 'age'        -> the head.* weights stored in best_model.pt
#   - other clocks -> best_head_<clock>.pt from the multi-clock run
# so one backbone forward pass per batch feeds all heads.

def available_clocks(head_path):
    """Clock names that have a trained head file (best_head_<clock>.pt) in `head_path`."""
    return sorted(p.stem[len("best_head_"):] for p in Path(head_path).glob("best_head_*.pt"))


def resolve_clocks(clocks, head_path):
    """
    Return the list of clocks to evaluate, always starting with 'age'.

    clocks=None means "'age' + every head found in head_path".
    """
    if clocks is None:
        clocks = available_clocks(head_path)
    clocks = [DEFAULT_CLOCK] + [c for c in clocks if c != DEFAULT_CLOCK]
    return list(dict.fromkeys(clocks))  # de-duplicate, keep order


def load_head_weights(clock, lora_ckpt, head_path):
    """
    Return (head_state_dict, meta) for one clock.

    meta is the head checkpoint's training info (None for 'age'), used to
    check the head was trained on the same LoRA backbone we are loading.
    """
    if clock == DEFAULT_CLOCK:
        ckpt  = torch.load(lora_ckpt, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        head  = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        if not head:
            raise KeyError(f"No 'head.*' keys in {lora_ckpt}; cannot load the '{clock}' head.")
        return head, None

    head_file = Path(head_path) / f"best_head_{clock}.pt"
    if not head_file.exists():
        raise FileNotFoundError(f"No head checkpoint for clock '{clock}' at {head_file}")
    ckpt = torch.load(head_file, map_location="cpu")
    return ckpt["head_state"], ckpt

def load_saved_scalers(head_path, scaler_file="scalers.pkl"):
    """
    Load the dict {clock: scaler} saved by the multi-clock training run, or
    return None if there is no scaler file.

    Pass the result to build_multiclock_dataloaders(scalers=...): clocks in
    the dict reuse their training scaler, and any clock not in it (typically
    'age') gets a new scaler fitted on this run's training split.
    """
    scaler_path = Path(head_path) / scaler_file
    if not scaler_path.exists():
        print(f"ℹ  No scaler file at {scaler_path}; all scalers will be fitted.")
        return None
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def ordered_loader(loader, batch_size=None):
    """
    Rebuild a DataLoader over the same dataset with shuffle=False, so batch
    order matches the split DataFrame's row order (the train loader shuffles).
    """
    from torch.utils.data import DataLoader
    ds = loader.dataset
    return DataLoader(ds, batch_size=batch_size or loader.batch_size,
                      shuffle=False, collate_fn=ds.collater, num_workers=0)


@torch.no_grad()
def predict_multiclock(model, loader, scalers, device, clock_names, sample_ids):
    """
    One backbone forward pass per batch, fanned out to every clock head.

    Same inference path as train.evaluate_multiclock (unmasked values,
    autocast, inverse-scaled outputs), but every sample is kept, even ones
    missing a label for some clock, and rows stay aligned to `sample_ids`.

    Returns
    -------
    DataFrame indexed by sample ID with '<clock>_pred' and '<clock>_true'
    columns (true is NaN where that clock has no label).
    """
    import pandas as pd
    model.eval()

    preds  = {n: [] for n in clock_names}
    truths = {n: [] for n in clock_names}

    for gene_ids, _, target_vals, ages, _, _ in loader:
        gene_ids    = gene_ids.to(device)
        target_vals = target_vals.to(device)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(gene_ids, target_vals, clock_names=clock_names)

        for name in clock_names:
            p = out[name].float().cpu().numpy().reshape(-1, 1)
            preds[name].append(scalers[name].inverse_transform(p).ravel())
            truths[name].append(ages[name].numpy())

    n = sum(len(b) for b in preds[clock_names[0]])
    if n != len(sample_ids):
        raise ValueError(f"Got {n} predictions for {len(sample_ids)} sample IDs; "
                         f"is the loader shuffled?")

    table = pd.DataFrame(index=pd.Index(sample_ids, name="ID"))
    for name in clock_names:
        table[f"{name}_pred"] = np.concatenate(preds[name])
        table[f"{name}_true"] = np.concatenate(truths[name])
    return table

# ─────────────────────────────────────────────────────────────────────────────
# Table Metrics Functions
# ─────────────────────────────────────────────────────────────────────────────

def cross_clock_metrics(predictions, clock_names, min_samples=2):
    """
    Compare every clock's predictions against every clock's true values.
 
    For each (pred clock i, true clock j) pair, uses all rows where both
    `<i>_pred` and `<j>_true` are non-NaN (pairwise, so a missing label for
    one clock doesn't drop that sample from other pairs).
 
    Parameters
    ----------
    predictions : DataFrame from predict_multiclock (or several concatenated),
                  with '<clock>_pred' and '<clock>_true' columns
    clock_names : list[str]
    min_samples : int — pairs with fewer usable samples are left as NaN
 
    Returns
    -------
    dict of three n×n DataFrames, keyed 'pearson_r', 'mae', 'r2'.
    Rows = predicted clock, columns = true clock, so the diagonal is each
    clock scored against its own labels. R² is r2_score(true, pred) and is
    not symmetric.
    """
 
    names = list(clock_names)
    out = {m: pd.DataFrame(np.nan, index=pd.Index(names, name="pred"),
                           columns=pd.Index(names, name="true"))
           for m in ("pearson_r", "mae", "r2")}
 
    for p in names:
        for t in names:
            pair = predictions[[f"{p}_pred", f"{t}_true"]].dropna()
            if len(pair) < min_samples:
                continue
            pred, true = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
 
            out["mae"].loc[p, t] = np.mean(np.abs(pred - true))
            # r and R² are undefined if either side is constant
            if np.std(pred) > 0 and np.std(true) > 0:
                out["pearson_r"].loc[p, t] = pearsonr(true, pred)[0]
                out["r2"].loc[p, t] = r2_score(true, pred)
    return out

def style_cross_clock(table, metric):
    """
    Apply a green-to-red heatmap to a cross-clock metric table.
    - pearson_r, r2 : higher is better (green = max)
    - mae           : lower is better (green = min)
    """
    high_is_good = metric in ("pearson_r", "r2")
    cmap = "RdYlGn" if high_is_good else "RdYlGn_r"
    return (
        table.style
             .background_gradient(cmap=cmap, axis=None)   # axis=None = across whole table
             .format("{:.3f}", na_rep="—")
             .set_caption(metric)
    )

# ─────────────────────────────────────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_clock_heatmap(cross, metric, mask_diagonal=True):
    table = cross[metric].astype(float)
    high_is_good = metric in ("pearson_r", "r2")
    cmap = "RdYlGn" if high_is_good else "RdYlGn_r"

    clean = [n.replace("_age", "") for n in table.index]
    table.index   = pd.Index(clean, name=table.index.name)
    table.columns = pd.Index(clean, name=table.columns.name)

    mask = np.eye(len(table), dtype=bool) if mask_diagonal else None

    fig, ax = plt.subplots(figsize=(len(table) + 3, len(table) + 3))
    sns.heatmap(
        table, annot=True, fmt=".2f", cmap=cmap,
        mask=mask, linewidths=0.5,
        ax=ax, vmin=table.values[~mask].min() if mask is not None else None,
    )
    ax.set_title(metric, pad=12)
    ax.set_xlabel("True clock"); ax.set_ylabel("Predicted clock")
    plt.tight_layout()
    return fig

def plot_true_clock_heatmap(predictions, clock_names, metric="pearson_r",
                             mask_diagonal=True, min_samples=2):

    names = list(clock_names)
    high_is_good = metric in ("pearson_r", "r2")
    cmap = "RdYlGn" if high_is_good else "RdYlGn_r"

    table = pd.DataFrame(np.nan, index=pd.Index(names, name="clock A"),
                                 columns=pd.Index(names, name="clock B"))
    for a in names:
        for b in names:
            pair = predictions[[f"{a}_true", f"{b}_true"]].dropna()
            if len(pair) < min_samples:
                continue
            va, vb = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
            if metric == "mae":
                table.loc[a, b] = np.mean(np.abs(va - vb))
            elif np.std(va) > 0 and np.std(vb) > 0:
                if metric == "pearson_r":
                    table.loc[a, b] = pearsonr(va, vb)[0]
                elif metric == "r2":
                    table.loc[a, b] = r2_score(vb, va)

    clean = [n.replace("_age", "") for n in names]
    table.index   = pd.Index(clean, name=table.index.name)
    table.columns = pd.Index(clean, name=table.columns.name)

    nan_mask = table.isna().values
    diag_mask = np.eye(len(names), dtype=bool) if mask_diagonal else np.zeros((len(names), len(names)), dtype=bool)
    mask = diag_mask | nan_mask

    off_diag = table.values[~diag_mask]
    vmin, vmax = np.nanmin(off_diag), np.nanmax(off_diag)

    fig, ax = plt.subplots(figsize=(len(names) + 3, len(names) + 3))
    sns.heatmap(
        table.fillna(0),
        annot=True, fmt=".2f", cmap=cmap,
        mask=mask, linewidths=0.5, ax=ax,
        vmin=vmin, vmax=vmax,
    )
    ax.set_title(f"Ground-truth pairwise {metric}", pad=12)
    plt.tight_layout()
    return fig
 
# Typographic / style constants matching NM figure conventions
_NM_RC = {
    # Font
    "font.family"        : "sans-serif",
    "font.sans-serif"    : ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size"          : 7,
    "axes.titlesize"     : 8,
    "axes.labelsize"     : 7,
    "xtick.labelsize"    : 6,
    "ytick.labelsize"    : 6,
    "legend.fontsize"    : 6,
    # Lines / markers
    "axes.linewidth"     : 0.75,
    "xtick.major.width"  : 0.75,
    "ytick.major.width"  : 0.75,
    "xtick.major.size"   : 3,
    "ytick.major.size"   : 3,
    "lines.linewidth"    : 1.0,
    # Layout
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "figure.dpi"         : 300,
    "savefig.dpi"        : 300,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.02,
}
 
# Colour palette
_SCATTER_FACE  = "#2166AC"   # blue fill
_SCATTER_EDGE  = "#1a1a1a"   # near-black edge
_IDENTITY_COL  = "#D6604D"   # red identity line
_REG_COL       = "#4D4D4D"   # dark-grey regression line
 
 
def _clean(name):
    """Strip common suffixes for axis / title labels."""
    return name.replace("_age", "").replace("_clock", "").replace("_", " ").title()
 
 
def _stat_annotation(ax, pearson_r, r2, mae, n, x=0.05, y=0.95):
    """
    Add a compact stats box in the upper-left corner, NM style (no box frame).
    """
    text = (
        f"r = {pearson_r:.2f}\n"
        f"R² = {r2:.2f}\n"
        f"MAE = {mae:.1f}\n"
        f"n = {n}"
    )
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top",
            linespacing=1.6, color="#1a1a1a")
 
 
def _scatter_panel(ax, x, y, xlabel, ylabel, title, identity=True, regression=True):
    """
    Draw one scatter panel: data points, optional identity and regression lines,
    return the computed stats dict.
    """
    # Points: thin dark edge + translucent blue fill
    ax.scatter(x, y, s=8, facecolor=_SCATTER_FACE, edgecolor=_SCATTER_EDGE,
               linewidths=0.3, alpha=0.75, zorder=3, rasterized=True)
 
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    pad  = (lims[1] - lims[0]) * 0.04
    lims = [lims[0] - pad, lims[1] + pad]
 
    if identity:
        ax.plot(lims, lims, color=_IDENTITY_COL, lw=0.9,
                ls="--", alpha=0.8, zorder=2, label="y = x")
 
    if regression:
        m, b   = np.polyfit(x, y, 1)
        reg_x  = np.array(lims)
        ax.plot(reg_x, m * reg_x + b, color=_REG_COL, lw=1.0,
                ls="-", alpha=0.9, zorder=2, label=f"slope = {m:.2f}")
 
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
 
    if identity or regression:
        ax.legend(frameon=False, handlelength=1.5, handletextpad=0.4)
 
    # Compute stats
    n  = len(x)
    r  = float(pearsonr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
    r2 = float(r2_score(y, x))    if np.std(x) > 0 and np.std(y) > 0 else np.nan
    mae = float(np.mean(np.abs(x - y)))
    return {"n": n, "pearson_r": r, "r2": r2, "mae": mae}
 
 
def plot_clock_panels(predictions, clock, chron_clock="age",
                      save_path=None, fig_width_mm=89):
    """
    Two-panel Nature Medicine-style figure for one epigenetic clock:
 
      Panel A — predicted <clock> vs. true <clock>
      Panel B — predicted <clock> vs. true chronological age
 
    Parameters
    ----------
    predictions  : DataFrame from predict_multiclock, containing
                   '<clock>_pred', '<clock>_true', '<chron_clock>_true'
    clock        : str — the clock to plot (e.g. 'horvath_age')
    chron_clock  : str — the chronological age clock column to use as x in
                   panel B (default 'age')
    save_path    : str | Path | None — if given, saves as PDF + PNG at 300 dpi
    fig_width_mm : float — NM single-column = 89 mm, double = 183 mm
 
    Returns
    -------
    fig, (ax_a, ax_b), stats
        stats is a dict with keys "pred_vs_true" and "pred_vs_chron",
        each containing n, pearson_r, r2, mae.
    """
    df = predictions[[f"{clock}_pred", f"{clock}_true",
                      f"{chron_clock}_true"]].dropna()
 
    if len(df) < 2:
        raise ValueError(
            f"Fewer than 2 samples with all three columns present for clock '{clock}'."
        )
 
    pred      = df[f"{clock}_pred"].to_numpy()
    true      = df[f"{clock}_true"].to_numpy()
    chron     = df[f"{chron_clock}_true"].to_numpy()
 
    label          = _clean(clock)
    chron_label    = _clean(chron_clock)
    fig_w_in       = fig_width_mm / 25.4
    fig_h_in       = fig_w_in * 0.52   # slightly wider than tall, NM convention
 
    with plt.rc_context(_NM_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(fig_w_in * 2, fig_h_in),
                                          constrained_layout=True)
 
        # ── Panel A: pred vs. true ───────────────────────────────────────────
        stats_a = _scatter_panel(
            ax_a, x=true, y=pred,
            xlabel=f"True {label} (years)",
            ylabel=f"Predicted {label} (years)",
            title=f"A   Predicted vs. true {label}",
        )
        _stat_annotation(ax_a, **stats_a)
 
        # ── Panel B: pred vs. chronological age ─────────────────────────────
        stats_b = _scatter_panel(
            ax_b, x=chron, y=pred,
            xlabel=f"Chronological age ({chron_label}, years)",
            ylabel=f"Predicted {label} (years)",
            title=f"B   Predicted {label} vs. chronological age",
            identity=True, regression=True,
        )
        _stat_annotation(ax_b, **stats_b)
 
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".svg"))
        print(f"Saved → {save_path}.svg")
 
    stats = {"pred_vs_true": stats_a, "pred_vs_chron": stats_b}
    return fig, (ax_a, ax_b), stats
 
 
def plot_all_clock_panels(predictions, clock_names, chron_clock="age",
                          save_dir=None, fig_width_mm=89):
    """
    Call plot_clock_panels for every clock in clock_names (skipping chron_clock
    itself, which has no separate head to evaluate).
 
    Returns
    -------
    dict[clock, (fig, axes, stats)]
    """
    results = {}
    clocks_to_plot = [c for c in clock_names if c != chron_clock]
 
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
 
    for clock in clocks_to_plot:
        save_path = (Path(save_dir) / clock) if save_dir else None
        try:
            fig, axes, stats = plot_clock_panels(
                predictions, clock,
                chron_clock   = chron_clock,
                save_path     = save_path,
                fig_width_mm  = fig_width_mm,
            )
            results[clock] = (fig, axes, stats)
            plt.show()
        except ValueError as e:
            print(f"⚠  Skipping '{clock}': {e}")
 
    return results

# ═════════════════════════════════════════════════════════════════════════════
# Missing-feature analysis (03_MissingFeatures.ipynb)
# ═════════════════════════════════════════════════════════════════════════════
 
def _mask_fraction(gene_ids, values, mask_frac, pad_id, pad_value=-2.0, mask_value=-1.0):
    """
    Randomly zero-out `mask_frac` of the non-padding CpG values in a batch
    by replacing them with `mask_value` (same convention as training).
 
    Parameters
    ----------
    gene_ids   : (batch, seq_len) int tensor
    values     : (batch, seq_len) float tensor — unmasked methylation values
    mask_frac  : float in [0, 1]
    pad_id     : int — token id used for padding
 
    Returns
    -------
    values_masked : tensor with same shape as values
    """
    if mask_frac == 0.0:
        return values.clone()
 
    non_pad = gene_ids.ne(pad_id)                   # (batch, seq_len) bool
    masked  = values.clone()
 
    # For each sample, randomly choose mask_frac of the non-padding positions
    for i in range(gene_ids.shape[0]):
        idx = non_pad[i].nonzero(as_tuple=True)[0]
        n_mask = max(1, int(len(idx) * mask_frac))
        chosen = idx[torch.randperm(len(idx))[:n_mask]]
        masked[i, chosen] = mask_value
 
    return masked
 
 
@torch.no_grad()
def evaluate_missing_fraction(model, loader, scalers, device, clock_names,
                               mask_fractions, pad_id, random_seed=42):
    """
    Run inference at a series of CpG masking levels and return per-clock
    performance at each level.
 
    For each mask fraction the SAME random seed is used so results at
    different fractions are comparable on the same samples.
 
    Parameters
    ----------
    model          : MultiClockPredictor, eval mode
    loader         : unshuffled DataLoader (use eu.ordered_loader)
    scalers        : dict[clock, scaler]
    device         : torch.device
    clock_names    : list[str]
    mask_fractions : list[float] — e.g. [0.0, 0.1, 0.2, ..., 0.9]
    pad_id         : int — model.pad_id
    random_seed    : int
 
    Returns
    -------
    DataFrame with columns [clock, mask_frac, mae, pearson_r, r2, n]
    """
    import pandas as pd
 
    rows = []
    for frac in mask_fractions:
        torch.manual_seed(random_seed)
 
        all_preds  = {n: [] for n in clock_names}
        all_labels = {n: [] for n in clock_names}
 
        for gene_ids, _, target_vals, ages, valid, _ in loader:
            gene_ids    = gene_ids.to(device)
            target_vals = target_vals.to(device)
 
            # Inject extra missingness on top of real missingness
            vals_masked = _mask_fraction(gene_ids, target_vals, frac, pad_id)
 
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(gene_ids, vals_masked, clock_names=clock_names)
 
            for name in clock_names:
                v = valid[name].numpy()
                if v.sum() == 0:
                    continue
                pred_norm = out[name].float().cpu().numpy()[v].reshape(-1, 1)
                pred_age  = scalers[name].inverse_transform(pred_norm).ravel()
                all_preds[name].extend(pred_age)
                all_labels[name].extend(ages[name].numpy()[v])
 
        for name in clock_names:
            p = np.array(all_preds[name])
            t = np.array(all_labels[name])
            n = len(p)
            if n < 2:
                rows.append(dict(clock=name, mask_frac=frac, mae=np.nan,
                                 pearson_r=np.nan, r2=np.nan, n=n))
                continue
            mae = float(np.mean(np.abs(p - t)))
            r   = float(pearsonr(p, t)[0]) if np.std(p) > 0 and np.std(t) > 0 else np.nan
            r2  = float(r2_score(t, p))    if np.std(p) > 0 else np.nan
            rows.append(dict(clock=name, mask_frac=frac, mae=mae,
                             pearson_r=r, r2=r2, n=n))
 
    return pd.DataFrame(rows)
 
 
def compute_sample_coverage(splits, clock_names, split="test"):
    """
    For every sample in `split`, compute the fraction of CpG values that are
    present (non-NaN) in the raw aligned data, then join the per-clock true
    labels so you can stratify performance by real observed coverage.
 
    Returns
    -------
    DataFrame indexed by sample ID with columns:
        cpg_coverage (float, 0–1), <clock>_true for each clock
    """
    import pandas as pd
 
    df = splits[f"{split}_df"].copy()
 
    # Each row's "data" is the aligned CpG array (vocab-ordered, NaN = missing)
    coverage = np.array([
        np.isfinite(row).mean() for row in df["data"]
    ])
 
    out = pd.DataFrame({"cpg_coverage": coverage}, index=df.index)
    for name in clock_names:
        if name in df.columns:
            out[f"{name}_true"] = df[name].values
    return out
 
 
def compute_coverage_performance(predictions, coverage_df, clock_names,
                                 n_bins=5, min_samples=3):
    """
    Bin samples by their real CpG coverage and compute per-clock MAE, Pearson r
    and R² within each bin. Only pred vs. true for the SAME clock.
 
    Parameters
    ----------
    predictions  : DataFrame from predict_multiclock
    coverage_df  : DataFrame from compute_sample_coverage
    clock_names  : list[str]
    n_bins       : int — number of equal-width coverage bins
    min_samples  : int — bins with fewer samples are dropped
 
    Returns
    -------
    DataFrame with columns [clock, coverage_bin, coverage_mid,
                             mae, pearson_r, r2, n]
    """
    import pandas as pd
 
    merged = predictions.join(coverage_df[["cpg_coverage"]], how="inner")
    bins   = np.linspace(0, 1, n_bins + 1)
    labels = [f"{bins[i]:.0%}–{bins[i+1]:.0%}" for i in range(n_bins)]
 
    merged["coverage_bin"] = pd.cut(merged["cpg_coverage"], bins=bins,
                                    labels=labels, include_lowest=True)
    merged["coverage_mid"] = pd.cut(merged["cpg_coverage"], bins=bins,
                                    labels=(bins[:-1] + bins[1:]) / 2,
                                    include_lowest=True).astype(float)
    rows = []
    for name in clock_names:
        for (bin_label, mid), grp in merged.groupby(
                ["coverage_bin", "coverage_mid"], observed=True):
            sub = grp[[f"{name}_pred", f"{name}_true"]].dropna()
            n   = len(sub)
            if n < min_samples:
                continue
            p, t = sub.iloc[:, 0].to_numpy(), sub.iloc[:, 1].to_numpy()
            mae  = float(np.mean(np.abs(p - t)))
            r    = float(pearsonr(p, t)[0]) if np.std(p) > 0 and np.std(t) > 0 else np.nan
            r2   = float(r2_score(t, p))    if np.std(p) > 0 else np.nan
            rows.append(dict(clock=name, coverage_bin=bin_label,
                             coverage_mid=mid, mae=mae, pearson_r=r, r2=r2, n=n))
 
    return pd.DataFrame(rows)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_missing_fraction_curves(mask_results, metric="mae", clock_names=None,
                                  ax=None):
    """
    Line plot of clock performance vs. injected CpG mask fraction.
    One line per clock.
    """
    if clock_names is None:
        clock_names = mask_results["clock"].unique().tolist()
 
    high_is_good = metric in ("pearson_r", "r2")
    ylabel = {"mae": "MAE (years)", "pearson_r": "Pearson r", "r2": "R²"}[metric]
 
    with plt.rc_context(_NM_RC):
        if ax is None:
            fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4))
        else:
            fig = ax.figure
 
        cmap  = plt.get_cmap("tab10")
        for i, name in enumerate(clock_names):
            sub = mask_results[mask_results["clock"] == name].sort_values("mask_frac")
            ax.plot(sub["mask_frac"] * 100, sub[metric], marker="o", markersize=3,
                    lw=1.0, color=cmap(i), label=_clean(name))
 
        ax.set_xlabel("Additional CpGs masked (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Clock performance vs. missing CpGs", pad=4, fontweight="bold")
        ax.legend(frameon=False, loc="best")
        if not high_is_good:
            ax.set_ylim(bottom=0)
        plt.tight_layout()
 
    return fig, ax
 
 
def plot_coverage_performance(coverage_results, metric="mae", clock_names=None):
    """
    Bar chart of per-clock performance within real CpG coverage bins.
    One subplot per clock, bars = coverage bins.
    """
    if clock_names is None:
        clock_names = coverage_results["clock"].unique().tolist()
 
    high_is_good = metric in ("pearson_r", "r2")
    ylabel = {"mae": "MAE (years)", "pearson_r": "Pearson r", "r2": "R²"}[metric]
    n_clocks = len(clock_names)
 
    with plt.rc_context(_NM_RC):
        fig_w = min(183, 50 * n_clocks) / 25.4
        fig, axes = plt.subplots(1, n_clocks, figsize=(fig_w, 60 / 25.4),
                                 sharey=True, constrained_layout=True)
        if n_clocks == 1:
            axes = [axes]
 
        cmap = plt.get_cmap("Blues")
 
        for ax, name in zip(axes, clock_names):
            sub = coverage_results[coverage_results["clock"] == name].sort_values("coverage_mid")
            if sub.empty:
                ax.set_title(_clean(name), pad=4, fontweight="bold")
                continue
            n_bars = len(sub)
            colors = [cmap(0.35 + 0.6 * i / max(n_bars - 1, 1)) for i in range(n_bars)]
            bars = ax.bar(sub["coverage_bin"], sub[metric], color=colors,
                          edgecolor="white", linewidth=0.4, width=0.6)
 
            # Sample-count labels above bars
            for bar, n in zip(bars, sub["n"]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (sub[metric].max() * 0.02),
                        f"n={n}", ha="center", va="bottom", fontsize=5)
 
            ax.set_title(_clean(name), pad=4, fontweight="bold")
            ax.set_xlabel("CpG coverage")
            ax.tick_params(axis="x", rotation=30)
            if not high_is_good:
                ax.set_ylim(bottom=0)
 
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"{ylabel} by real CpG coverage", fontsize=8, fontweight="bold")
 
    return fig, axes
 