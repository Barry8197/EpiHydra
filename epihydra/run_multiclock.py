"""
EpiHydra multi-clock deployment script.

Trains one independent AttentionPoolingHead per epigenetic clock on top of a
single FROZEN backbone. The backbone is NEVER trained here — there is no
optimiser group for it and no unfreezing path.

The backbone may optionally carry LoRA adapters that were trained earlier by
run.py. Those adapters are LOADED and FROZEN, not trained: they are part of
the fixed representation, exactly as in the train_epi_head notebook where the
PEFT-wrapped backbone is restored from a checkpoint and then has
requires_grad set to False.

Each head is a separate model that shares a cached representation:

  * the backbone forward pass runs ONCE per batch (MultiClockPredictor.encode,
    always under no_grad), then fans out to every head
  * no head shares a trainable parameter with any other head, so the summed
    loss in train_one_epoch_multiclock yields exactly the gradients each head
    would receive if trained alone
  * every head is selected, checkpointed and scored on ITS OWN validation MAE.
    Nothing is averaged across clocks.

Usage
-----
# frozen pretrained backbone, no adapters
python run_multiclock.py \\
    --data_path  /path/to/methylation.pkl \\
    --model_dir  pretrained_models/tiny \\
    --clocks     hannum_age horvath_age phenoage

# frozen LoRA-adapted backbone from a previous run.py checkpoint
python run_multiclock.py \\
    --data_path  /path/to/methylation.pkl \\
    --model_dir  pretrained_models/tiny \\
    --lora_ckpt  .../checkpoints/nlayers6_nhead4_emb64_.../best_model.pt \\
    --clocks     hannum_age horvath_age phenoage

Checkpoint layout written per run
---------------------------------
<save_dir>/<run_id>/
    scalers.pkl              <- dict[clock, MinMaxScaler], needed at inference
    best_head_<clock>.pt     <- one file per clock, best epoch for that clock
"""
import warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*nested tensor.*")

import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from methylgpt import MethylGPTModel, MethylVocab
from scgpt.model.model import TransformerModel

from epihydra import ( 
    VocabWrapper, 
    MultiClockPredictor, 
    load_backbone,
    build_multiclock_dataloaders,
    train_one_epoch_multiclock,
    evaluate_multiclock,
)

# Must match the target_modules used when the adapters were trained (run.py)
LORA_TARGET_MODULES = ["Wqkv", "out_proj", "linear1", "linear2"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="EpiHydra: train one independent clock head per clock on a "
                    "frozen (optionally LoRA-adapted) backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Paths")
    g.add_argument("--data_path", required=True, type=Path,
                   help="Path to methylation.pkl (dict with 'meta' and 'CpG')")
    g.add_argument("--model_dir", required=True, type=Path,
                   help="Directory containing args.json, vocab.json and *.pt")
    g.add_argument("--log_dir",   default=Path("runs"),        type=Path)
    g.add_argument("--save_dir",  default=Path("checkpoints"), type=Path)

    g = p.add_argument_group("Frozen LoRA backbone (loaded, never trained)")
    g.add_argument("--lora_ckpt", default=None, type=Path,
                   help="run.py checkpoint (best_model.pt) whose trained LoRA "
                        "adapters are loaded into the backbone and frozen. "
                        "Omit to use the plain pretrained backbone.")
    g.add_argument("--lora_r",     default=None, type=int,
                   help="LoRA rank. Default: read from checkpoint run_config, else 8")
    g.add_argument("--lora_alpha", default=None, type=int,
                   help="LoRA alpha. Default: read from checkpoint run_config, else 4")
    g.add_argument("--keep_adapter", action="store_true",
                   help="Keep the PEFT wrapper live instead of merging adapters "
                        "into the base weights. Merging is default: it is "
                        "numerically equivalent, faster, and avoids any "
                        "adapter-bypass edge cases in eval mode.")
    g.add_argument("--init_heads_from_ckpt", action="store_true",
                   help="Warm-start every clock head from the single head stored "
                        "in --lora_ckpt instead of random init")

    g = p.add_argument_group("Clocks")
    g.add_argument("--clocks", required=True, nargs="+", metavar="NAME",
                   help="Column names in data['meta'], e.g. "
                        "--clocks hannum_age horvath_age phenoage")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       default=50,   type=int)
    g.add_argument("--batch_size",   default=128,  type=int)
    g.add_argument("--lr_head",      default=1e-4, type=float)
    g.add_argument("--weight_decay", default=0.01, type=float)
    g.add_argument("--mask_ratio",   default=0.15, type=float,
                   help="Upper bound on random CpG masking in the train split")
    g.add_argument("--num_workers",  default=4,    type=int)

    g = p.add_argument_group("Model")
    g.add_argument("--num_heads", default=4,   type=int)
    g.add_argument("--dropout",   default=0.1, type=float)

    g = p.add_argument_group("Data splits")
    g.add_argument("--train_ratio",  default=0.70, type=float)
    g.add_argument("--valid_ratio",  default=0.15, type=float)
    g.add_argument("--test_ratio",   default=0.15, type=float)
    g.add_argument("--random_state", default=42,   type=int)

    g = p.add_argument_group("Logging")
    g.add_argument("--log_every", default=0, type=int,
                   help="Log attention weight histograms every N epochs (0 = off)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model dir validation
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_FILES = {
    "args.json"  : "pretrain config",
    "vocab.json" : "pretrained vocab (probe_id -> token_id)",
}

def validate_model_dir(model_dir):
    if not model_dir.exists():
        print(f"\n[EpiHydra] ERROR — model_dir not found:\n  {model_dir}\n")
        raise SystemExit(1)

    errors = []
    for filename, description in _REQUIRED_FILES.items():
        if not (model_dir / filename).exists():
            errors.append(f"  missing {filename:<20} ({description})")

    pt_files = list(model_dir.glob("*.pt"))
    if not pt_files:
        errors.append("  missing *.pt                 (model weights)")

    if errors:
        print(f"\n[EpiHydra] ERROR — model_dir is missing required files:")
        print(f"  model_dir : {model_dir}\n")
        for e in errors:
            print(e)
        print(f"\n  Expected layout:\n  {model_dir}/\n    args.json\n    vocab.json\n    <weights>.pt\n")
        raise SystemExit(1)

    return {
        "config"  : model_dir / "args.json",
        "vocab"   : model_dir / "vocab.json",
        "weights" : str(pt_files[0]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Frozen LoRA backbone loading
# ──────────────────────────────────────────────────────────────────────────────

def load_lora_adapters(backbone, ckpt_path, device, lora_r, lora_alpha, merge=True):
    """
    Rebuild the PEFT wrapper used by run.py, restore trained adapter weights
    from its checkpoint, verify they actually landed, then (by default) merge
    them into the base weights.

    The checkpoint is an AgePredictor state_dict, so backbone keys carry a
    'backbone.' prefix. The stored head is ignored here — see
    --init_heads_from_ckpt to reuse it.

    Returns
    -------
    backbone : nn.Module — merged plain module, or the live PEFT wrapper
    info     : dict      — what was loaded, for logging
    """
    from peft import LoraConfig, get_peft_model

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    rc    = ckpt.get("run_config", {}) or {}

    r     = lora_r     if lora_r     is not None else int(rc.get("lora_r", 8))
    alpha = lora_alpha if lora_alpha is not None else int(rc.get("lora_alpha", 4))

    print(f"\nLoading frozen LoRA adapters from {ckpt_path}")
    if "run_id" in ckpt:
        print(f"  source run   : {ckpt['run_id']}")
    if "epoch" in ckpt:
        print(f"  source epoch : {ckpt['epoch']}  (valid MAE {ckpt.get('mae', float('nan')):.3f})")
    print(f"  lora_r={r} lora_alpha={alpha} targets={LORA_TARGET_MODULES}")

    config = LoraConfig(
        r              = r,
        lora_alpha     = alpha,
        target_modules = LORA_TARGET_MODULES,
        lora_dropout   = 0.1,
    )
    backbone = get_peft_model(backbone, config)

    backbone_state = {
        k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
    }
    if not backbone_state:
        print("\n[EpiHydra] ERROR — no 'backbone.*' keys in checkpoint. Is this a "
              "run.py AgePredictor checkpoint?\n")
        raise SystemExit(1)

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)

    # ── Verify the adapters are real, not silently absent ────────────────────
    ckpt_lora    = [k for k in backbone_state if "lora_" in k]
    missing_lora = [k for k in missing        if "lora_" in k]
    unexp_lora   = [k for k in unexpected     if "lora_" in k]

    # lora_B initialises to zero; a trained adapter must have non-zero B.
    b_norms = [
        float(v.abs().sum()) for k, v in backbone.state_dict().items()
        if "lora_B" in k
    ]
    n_nonzero_b = sum(1 for n in b_norms if n > 0)

    print(f"  adapter tensors in checkpoint : {len(ckpt_lora)}")
    print(f"  adapter tensors in model      : {len(b_norms) * 2} "
          f"(lora_A + lora_B pairs)")
    print(f"  lora_B with non-zero weights  : {n_nonzero_b}/{len(b_norms)}")
    if missing_lora:
        print(f"  ⚠  {len(missing_lora)} adapter tensor(s) in the model were NOT "
              f"found in the checkpoint, e.g. {missing_lora[0]}")
    if unexp_lora:
        print(f"  ⚠  {len(unexp_lora)} adapter tensor(s) in the checkpoint have no "
              f"home in this model, e.g. {unexp_lora[0]}")
        print(f"     LoRA config here probably differs from the one used to train it.")
    if n_nonzero_b == 0:
        print(f"  ⚠  Every lora_B is zero — the adapters are an identity map and "
              f"this backbone is numerically identical to the plain pretrained "
              f"one. Check that the checkpoint really contains trained adapters.")

    non_lora_missing = [k for k in missing if "lora_" not in k]
    if non_lora_missing:
        print(f"  note: {len(non_lora_missing)} non-adapter key(s) not in checkpoint "
              f"(kept from pretrained weights)")

    info = {
        "ckpt"            : str(ckpt_path),
        "source_run_id"   : ckpt.get("run_id"),
        "source_epoch"    : ckpt.get("epoch"),
        "lora_r"          : r,
        "lora_alpha"      : alpha,
        "target_modules"  : LORA_TARGET_MODULES,
        "n_adapter_ckpt"  : len(ckpt_lora),
        "n_lora_B_nonzero": n_nonzero_b,
        "n_lora_B_total"  : len(b_norms),
        "merged"          : merge,
    }

    if merge:
        backbone = backbone.merge_and_unload()
        print(f"  adapters merged into base weights\n")
    else:
        print(f"  PEFT wrapper kept live (--keep_adapter)\n")

    return backbone.to(device), info


def load_head_state(ckpt_path):
    """Pull the single AgePredictor head state_dict out of a run.py checkpoint."""
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    head  = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    if not head:
        print("[EpiHydra] ERROR — no 'head.*' keys in checkpoint; cannot warm-start.")
        raise SystemExit(1)
    return head


# ──────────────────────────────────────────────────────────────────────────────
# Run ID
# ──────────────────────────────────────────────────────────────────────────────

def build_run_id(args, pretrain_config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        f"clocks{len(args.clocks)}",
        f"nlayers{pretrain_config['nlayers']}",
        f"nhead{pretrain_config['nhead']}",
        f"emb{pretrain_config['layer_size']}",
        "lora" if args.lora_ckpt else "nolora",
        f"lr{args.lr_head:.0e}",
        f"bs{args.batch_size}",
        f"mask{args.mask_ratio}",
        f"heads{args.num_heads}",
        timestamp,
    ]
    return "_".join(parts)


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


# ──────────────────────────────────────────────────────────────────────────────
# Attention logging
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def log_attention_weights(model, loader, device, writer, epoch, clock_names):
    """
    Log per-clock attention weight histograms.

    NOTE: attention is logged as a distribution only. Position indices are NOT
    reported, because pad_batch subsamples and reorders probes per sample, so
    column j refers to a different CpG in every row. Mapping attention back to
    probes requires scattering through gene_ids into vocab space.
    """
    model.eval()

    gene_ids, _, target_vals, _, _, _ = next(iter(loader))
    gene_ids    = gene_ids.to(device)
    target_vals = target_vals.to(device)

    sequence_hidden, mask = model.encode(gene_ids, target_vals)

    for name in clock_names:
        _, attn_weights = model.heads[name](sequence_hidden, key_padding_mask=mask)
        writer.add_histogram(
            f"attention/{name}/weights", attn_weights.float().flatten(), epoch
        )


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def scatter_per_clock(name, result, run_id, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    preds, labels = result["preds"], result["labels"]
    if len(preds) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, s=15, c="black",     alpha=1.0, zorder=1)
    ax.scatter(labels, preds, s=10, c="steelblue", alpha=0.5, zorder=2)
    lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
    ax.plot(lims, lims, "r--", alpha=0.8, label="y = x")
    ax.set_xlabel(f"True {name} (years)")
    ax.set_ylabel(f"Predicted {name} (years)")
    ax.set_title(f"{name} | {run_id[:40]}\nMAE={result['mae']:.1f} yrs | n={len(preds)}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clock_names = list(dict.fromkeys(args.clocks))

    if args.init_heads_from_ckpt and args.lora_ckpt is None:
        print("[EpiHydra] ERROR — --init_heads_from_ckpt requires --lora_ckpt")
        raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"  EpiHydra Multi-Clock Head Training")
    print(f"  Device   : {device}")
    print(f"  Clocks   : {', '.join(clock_names)}")
    print(f"  Backbone : {'LoRA-adapted (frozen)' if args.lora_ckpt else 'pretrained (frozen)'}")
    print(f"{'='*60}\n")

    model_paths = validate_model_dir(args.model_dir)
    print(f"Model dir : {args.model_dir}")
    print(f"  config  : {model_paths['config'].name}")
    print(f"  vocab   : {model_paths['vocab'].name}")
    print(f"  weights : {Path(model_paths['weights']).name}\n")

    with open(model_paths["config"]) as f:
        pretrain_config = json.load(f)

    run_id   = build_run_id(args, pretrain_config)
    log_path = args.log_dir  / run_id
    ckpt_dir = args.save_dir / run_id
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID   : {run_id}")
    print(f"Logs     : {log_path}")
    print(f"Checkpts : {ckpt_dir}\n")

    writer = SummaryWriter(log_dir=str(log_path))

    run_config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        **pretrain_config,
        "run_id"           : run_id,
        "device"           : str(device),
        "clock_names"      : clock_names,
        "backbone_trained" : False,
    }

    print("Loading data...")
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    vocab = VocabWrapper(model_paths["vocab"], pretrain_config["n_hvg"])

    splits = build_multiclock_dataloaders(
        data         = data,
        vocab        = vocab,
        clock_names  = clock_names,
        train_ratio  = args.train_ratio,
        valid_ratio  = args.valid_ratio,
        test_ratio   = args.test_ratio,
        mask_ratio   = args.mask_ratio,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        random_state = args.random_state,
        verbose      = True,
    )
    train_loader = splits["train_loader"]
    valid_loader = splits["valid_loader"]
    test_loader  = splits["test_loader"]
    scalers      = splits["scalers"]

    writer.add_text("data/cpg_coverage",
        f"Present: {splits['n_present_cpgs']} | Missing: {splits['n_missing_cpgs']}")
    writer.add_text("data/labels_per_clock",
        " | ".join(f"{k}: {v}" for k, v in splits["n_present_per_clock"].items()))
    for name in clock_names:
        s = scalers[name]
        writer.add_text(f"data/scaler_range/{name}",
            f"{name} range seen during training: "
            f"{s.data_min_[0]:.1f} – {s.data_max_[0]:.1f} years")

    with open(ckpt_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    # ── Backbone ─────────────────────────────────────────────────────
    backbone = load_backbone(model_paths["weights"], vocab, pretrain_config)

    lora_info = None
    if args.lora_ckpt is not None:
        backbone, lora_info = load_lora_adapters(
            backbone, args.lora_ckpt, device,
            lora_r     = args.lora_r,
            lora_alpha = args.lora_alpha,
            merge      = not args.keep_adapter,
        )
        run_config["lora"] = lora_info
        writer.add_text("model/lora", json.dumps(lora_info, indent=2))

    # Frozen for the whole run — adapters included. MultiClockPredictor.encode
    # runs under no_grad regardless; this makes it explicit and keeps
    # untrainable tensors out of the optimiser.
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    with open(log_path / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    pad_id = vocab.vocab["<pad>"]
    model  = MultiClockPredictor(
        backbone,
        embed_dim   = pretrain_config["layer_size"],
        pad_id      = pad_id,
        clock_names = clock_names,
        num_heads   = args.num_heads,
        dropout     = args.dropout,
    ).to(device)

    if args.init_heads_from_ckpt:
        head_state = load_head_state(args.lora_ckpt)
        for name in clock_names:
            model.heads[name].load_state_dict(head_state)
        print(f"Warm-started all {len(clock_names)} heads from the checkpoint head\n")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_per_head  = sum(p.numel() for p in model.heads[clock_names[0]].parameters())
    print(f"Total params      : {n_total:,}")
    print(f"Trainable params  : {n_trainable:,}  ({len(clock_names)} heads, backbone frozen)")
    print(f"Params per head   : {n_per_head:,}")
    assert n_trainable == n_per_head * len(clock_names), \
        "Something outside the heads is trainable — backbone freeze failed"
    print()

    writer.add_text("model/architecture", str(model))
    writer.add_text("model/params",
        f"Total: {n_total:,} | Trainable: {n_trainable:,} | Per head: {n_per_head:,}")

    # ── Optimiser ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [{"params": model.heads[name].parameters(), "lr": args.lr_head}
         for name in clock_names],
        weight_decay=args.weight_decay,
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    scaler_amp = torch.cuda.amp.GradScaler()

    # ── Training loop ────────────────────────────────────────────────
    best = {name: {"mae": float("inf"), "epoch": 0, "ckpt": None} for name in clock_names}

    print(f"Training {len(clock_names)} independent heads for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):

        train_losses = train_one_epoch_multiclock(
            model, train_loader, optimizer, scaler_amp, device, clock_names,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        valid_results = evaluate_multiclock(
            model, valid_loader, scalers, device, clock_names,
        )

        writer.add_scalar("lr", current_lr, epoch)

        lines = []
        for name in clock_names:
            loss = train_losses[name]
            mae  = valid_results[name]["mae"]

            if not np.isnan(loss):
                writer.add_scalar(f"loss/train/{name}", loss, epoch)
            if not np.isnan(mae):
                writer.add_scalar(f"mae/valid/{name}", mae, epoch)

            improved = ""
            if not np.isnan(mae) and mae < best[name]["mae"]:
                ckpt_path = ckpt_dir / f"best_head_{safe_filename(name)}.pt"
                torch.save(
                    {
                        "clock"      : name,
                        "epoch"      : epoch,
                        "run_id"     : run_id,
                        "head_state" : model.heads[name].state_dict(),
                        "valid_mae"  : mae,
                        "scaler"     : scalers[name],
                        "lora"       : lora_info,
                        "run_config" : run_config,
                    },
                    ckpt_path,
                )
                best[name] = {"mae": mae, "epoch": epoch, "ckpt": ckpt_path}
                improved = " *"

            mae_str  = "  nan" if np.isnan(mae)  else f"{mae:.2f}"
            loss_str = "  nan" if np.isnan(loss) else f"{loss:.4f}"
            lines.append(
                f"    {name:<20} loss {loss_str} | valid MAE {mae_str} yrs | "
                f"best {best[name]['mae']:.2f} (ep {best[name]['epoch']}){improved}"
            )

        if args.log_every and epoch % args.log_every == 0:
            log_attention_weights(model, valid_loader, device, writer, epoch, clock_names)

        print(f"Epoch {epoch:3d}/{args.epochs} | LR: {current_lr:.2e}")
        print("\n".join(lines))

    print(f"\n{'─'*56}")
    print("  Best validation MAE per clock")
    print(f"{'─'*56}")
    for name in clock_names:
        print(f"  {name:<20} {best[name]['mae']:>8.2f} yrs  (epoch {best[name]['epoch']})")
    print(f"{'─'*56}\n")

    # ── Test evaluation, each clock at its own best epoch ────────────
    print("Evaluating each clock at its own best epoch...\n")

    metrics = {"run_id": run_id, "clock_names": clock_names,
               "lora": lora_info, "clocks": {}}

    print(f"{'─'*64}")
    print(f"  {'clock':<20} {'epoch':>6} {'n':>6} {'MAE':>8} {'R²':>8} {'r':>8}")
    print(f"{'─'*64}")

    for name in clock_names:
        if best[name]["ckpt"] is None:
            print(f"  {name:<20} {'—':>6} {0:>6} {'nan':>8} {'nan':>8} {'nan':>8}")
            metrics["clocks"][name] = {
                "note": "no labeled validation samples — head never checkpointed",
                "best_epoch": None,
            }
            continue

        ckpt = torch.load(best[name]["ckpt"], map_location=device)
        model.heads[name].load_state_dict(ckpt["head_state"])

        result = evaluate_multiclock(model, test_loader, scalers, device, [name])[name]
        preds, labels, mae = result["preds"], result["labels"], result["mae"]

        if len(preds) >= 2 and np.std(labels) > 0 and np.std(preds) > 0:
            r2   = float(r2_score(labels, preds))
            r, _ = pearsonr(labels, preds)
            r    = float(r)
        else:
            r2, r = float("nan"), float("nan")

        print(f"  {name:<20} {best[name]['epoch']:>6} {len(preds):>6} "
              f"{mae:>8.2f} {r2:>8.3f} {r:>8.3f}")

        if not np.isnan(mae):
            writer.add_scalar(f"mae/test/{name}", mae, 0)
        if not np.isnan(r2):
            writer.add_scalar(f"r2/test/{name}",        r2, 0)
            writer.add_scalar(f"pearson_r/test/{name}", r,  0)

        metrics["clocks"][name] = {
            "best_epoch"     : best[name]["epoch"],
            "best_valid_mae" : best[name]["mae"],
            "checkpoint"     : str(best[name]["ckpt"]),
            "test_n"         : int(len(preds)),
            "test_mae"       : mae,
            "test_r2"        : r2,
            "test_pearson_r" : r,
        }

        fig = scatter_per_clock(name, result, run_id,
                                log_path / f"scatter_{safe_filename(name)}.png")
        if fig is not None:
            writer.add_figure(f"test/scatter/{name}", fig)
            import matplotlib.pyplot as plt
            plt.close(fig)

    print(f"{'─'*64}\n")

    with open(log_path / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    writer.flush()
    writer.close()

    print(f"Logs        : {log_path}")
    print(f"Checkpoints : {ckpt_dir}")
    for name in clock_names:
        if best[name]["ckpt"] is not None:
            print(f"  {name:<20} -> {best[name]['ckpt'].name}")
    print(f"\nTo view all runs:\n  tensorboard --logdir {args.log_dir}\n")


if __name__ == "__main__":
    main()