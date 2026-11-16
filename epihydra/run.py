"""
EpiHydra cluster deployment script.

Each invocation creates a unique run identifier from model args + timestamp.
All runs write to the same --log_dir so a single TensorBoard instance shows
every run as a separate entry.

Usage
-----
python run.py \\
    --data_path  /path/to/methylation.pkl \\
    --model_dir  pretrained_models/tiny \\
    --log_dir    /shared/runs \\
    --save_dir   /shared/checkpoints \\
    --epochs     50 \\
    --batch_size 128

Expected model_dir layout
--------------------------
pretrained_models/tiny/
    args.json       ← pretrain config  (required)
    vocab.json      ← probe → token ID (required)
    *.pt            ← model weights    (required)

Viewing all runs
----------------
tensorboard --logdir /shared/runs
"""
import warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*nested tensor.*")

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from scgpt.model.model import TransformerModel

from peft import LoraConfig, get_peft_model

from epihydra import (
    VocabWrapper,
    AgePredictor,
    LinearProbeHead,
    init_head_from_ridge,
    load_backbone,
    build_dataloaders,
    train_one_epoch,
    evaluate,
    precompute_cpg_ranks,
    DEFAULT_MASK_WEIGHTS
)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="EpiHydra: finetune MethylGPT for age prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Paths")
    g.add_argument("--data_path", required=True, type=Path,
                   help="Path to methylation.pkl (dict with 'meta' and 'CpG')")
    g.add_argument("--model_dir", required=True, type=Path,
                   help="Directory containing args.json, vocab.json and *.pt")
    g.add_argument("--log_dir",   default=Path("runs"),        type=Path,
                   help="Root TensorBoard log directory (shared across runs)")
    g.add_argument("--save_dir",  default=Path("checkpoints"), type=Path,
                   help="Root checkpoint directory (shared across runs)")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       default=50,   type=int,
                   help="Stage-2 epochs (head + LoRA)")
    g.add_argument("--head_epochs",  default=0,    type=int,
                   help="Stage-1 epochs training the head only, LoRA frozen "
                        "(0 = single-stage training, as before)")
    g.add_argument("--batch_size",   default=128,  type=int)
    g.add_argument("--lr_head",      default=1e-4, type=float)
    g.add_argument("--lr_lora",      default=1e-4, type=float)
    g.add_argument("--lora_r",       default=8,    type=int)
    g.add_argument("--lora_alpha",   default=4,    type=int)
    g.add_argument("--weight_decay", default=0.01, type=float)
    g.add_argument("--clip_grad",    default=1.0,  type=float)
    g.add_argument("--mask_ratio",   default=0.15, type=float,
                   help="Fraction of CpG values masked per batch")
    g.add_argument("--num_workers",  default=4,    type=int)

    g = p.add_argument_group("Model")
    g.add_argument("--num_heads", default=4,   type=int,
                   help="Attention heads in pooling head")
    g.add_argument("--n_queries", default=1,   type=int,
                   help="Number of learned query vectors in attention pooling head")
    g.add_argument("--dropout",   default=0.1, type=float)
    g.add_argument("--head_type", default="probe", choices=["probe", "attention"],
                   help="probe = ridge-initialised linear head; attention = original pooling head")

    g = p.add_argument_group("Data splits")
    g.add_argument("--train_ratio",  default=0.70, type=float)
    g.add_argument("--valid_ratio",  default=0.15, type=float)
    g.add_argument("--test_ratio",   default=0.15, type=float)
    g.add_argument("--random_state", default=42,   type=int)

    g = p.add_argument_group("Logging")
    g.add_argument("--log_every", default=10, type=int,
                   help="Log attention weight histograms every N epochs")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model dir validation
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_FILES = {
    "args.json"  : "pretrain config",
    "vocab.json" : "pretrained vocab (probe_id → token_id)",
}

def validate_model_dir(model_dir):
    """
    Verify model_dir exists and contains all required files.

    Returns
    -------
    dict — resolved paths for 'config', 'vocab', 'weights'

    Raises
    ------
    SystemExit with a clear message listing every missing item.
    """
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
        print(f"\n  Expected layout:")
        print(f"  {model_dir}/")
        print(f"    args.json")
        print(f"    vocab.json")
        print(f"    <weights>.pt\n")
        raise SystemExit(1)

    return {
        "config"  : model_dir / "args.json",
        "vocab"   : model_dir / "vocab.json",
        "weights" : str(pt_files[0]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Run ID
# ──────────────────────────────────────────────────────────────────────────────

def build_run_id(args, pretrain_config):
    """
    Unique, human-readable run identifier.
 
    Format: nlayers{N}_nhead{N}_emb{N}_lora{r}a{alpha}_lr{lr}_heads{h}_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        f"nlayers{pretrain_config['nlayers']}",
        f"nhead{pretrain_config['nhead']}",
        f"emb{pretrain_config['layer_size']}",
        f"lora{args.lora_r}a{args.lora_alpha}",
        f"lr{args.lr_lora:.0e}",
        f"heads{args.num_heads}",
        timestamp,
    ]
    return "_".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Attention logging
# ──────────────────────────────────────────────────────────────────────────────

# def log_attention_weights(model, loader, pad_id, device, writer, epoch):
#     """Log per-position attention weight histogram to TensorBoard."""
#     model.eval()
#     with torch.no_grad():
#         gene_ids, _, target_vals, _, _ = next(iter(loader))
#         gene_ids    = gene_ids.to(device)
#         target_vals = target_vals.to(device)
#         mask        = gene_ids.eq(pad_id)

#         hook = model.backbone.transformer_encoder.register_forward_hook(model._hook_fn)
#         _ = model.backbone(
#             gene_ids, target_vals,
#             src_key_padding_mask = mask,
#             batch_labels         = None,
#         )
#         hook.remove()

#         _, attn_weights = model.head(model._sequence_hidden, key_padding_mask=mask)
#         mean_weights = attn_weights.mean(dim=(0, 1)).squeeze()  # (seq_len,)

#         writer.add_histogram("attention/mean_weights", mean_weights, epoch)

#         top_vals, top_idx = mean_weights.topk(10)
#         top_str = " | ".join(
#             [f"pos {i.item()}: {v.item():.4f}" for i, v in zip(top_idx, top_vals)]
#         )
#         writer.add_text("attention/top10_positions", top_str, epoch)


# ──────────────────────────────────────────────────────────────────────────────
# Recalibrate Batch Norm
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def recalibrate_bn(model, loader, device, n_batches=40):
    """Re-estimate the head's BatchNorm running stats with the current weights."""
    bns = [m for m in model.head.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                     # cumulative average over the batches below
    model.eval()                               # backbone stays in eval mode (no dropout)
    model.head.train()                         # only the head's BatchNorm collects statistics
    for i, (gene_ids, _, target_vals, *_) in enumerate(loader):
        if i >= n_batches:
            break
        with torch.cuda.amp.autocast():
            model(gene_ids.to(device), target_vals.to(device))
    for bn in bns:
        bn.momentum = 0.1

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  EpiHydra Age Prediction")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Validate model_dir ───────────────────────────────────────────
    model_paths = validate_model_dir(args.model_dir)
    print(f"Model dir : {args.model_dir}")
    print(f"  config  : {model_paths['config'].name}")
    print(f"  vocab   : {model_paths['vocab'].name}")
    print(f"  weights : {Path(model_paths['weights']).name}\n")

    # ── Load pretrain config ─────────────────────────────────────────
    with open(model_paths["config"]) as f:
        pretrain_config = json.load(f)

    # ── Build unique run ID ──────────────────────────────────────────
    run_id   = build_run_id(args, pretrain_config)
    log_path = args.log_dir  / run_id
    ckpt_dir = args.save_dir / run_id
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID   : {run_id}")
    print(f"Logs     : {log_path}")
    print(f"Checkpts : {ckpt_dir}\n")

    # ── TensorBoard writer ───────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(log_path))

    run_config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        **pretrain_config,
        "run_id": run_id,
        "device": str(device),
    }
    with open(log_path / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading data...")
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    # ── Build vocab ──────────────────────────────────────────────────
    vocab = VocabWrapper(model_paths["vocab"], pretrain_config["n_hvg"])

    # ── model_args — passed through to precompute_tokens ────────────
    model_args = {
        "n_hvg"      : pretrain_config["n_hvg"],
        "mask_ratio" : args.mask_ratio,
        "mask_value" : vocab.mask_value,   # -1, defined in VocabWrapper
    }

    # ── Build dataloaders ────────────────────────────────────────────
    splits = build_dataloaders(
        data         = data,
        vocab        = vocab,
        train_ratio  = args.train_ratio,
        valid_ratio  = args.valid_ratio,
        test_ratio   = args.test_ratio,
        batch_size   = args.batch_size,
        mask_ratio   = args.mask_ratio,
        num_workers  = args.num_workers,
        random_state = args.random_state,
        verbose      = True,
    )
    train_loader = splits["train_loader"]
    valid_loader = splits["valid_loader"]
    test_loader  = splits["test_loader"]
    scaler       = splits["scaler"]
    cpg_ranks = precompute_cpg_ranks(splits["train_df"])

    writer.add_text("data/cpg_coverage",
        f"Present: {splits['n_present_cpgs']} | Missing: {splits['n_missing_cpgs']}")
    writer.add_text("data/scaler_range",
        f"Age range seen during training: "
        f"{scaler.data_min_[0]:.1f} – {scaler.data_max_[0]:.1f} years")

    with open(ckpt_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── Build model ──────────────────────────────────────────────────
    backbone  = load_backbone(model_paths["weights"], vocab, pretrain_config)
    pad_id    = vocab.vocab["<pad>"]
    age_model = AgePredictor(
        backbone,
        embed_dim = pretrain_config["layer_size"],
        pad_id    = pad_id,
        num_heads = args.num_heads,
        dropout   = args.dropout,
        n_queries = args.n_queries,
    ).to(device)

    if args.head_type == "probe":
        age_model.head = LinearProbeHead(pretrain_config["layer_size"]).to(device)
        init_head_from_ridge(age_model, train_loader, scaler, device, pad_id)
        recalibrate_bn(age_model, train_loader, device)
        probe_mae, *_ = evaluate(age_model, valid_loader, scaler, device)
        print(f"Head initialised from ridge probe | valid MAE {probe_mae:.2f} yrs")

    config = LoraConfig(
        r=args.lora_r,                              
        lora_alpha=args.lora_alpha,
        target_modules=["Wqkv", "out_proj","linear1" , "linear2"], 
        lora_dropout=0.1,
    )
    
    age_model.backbone = get_peft_model(age_model.backbone, config)
    print(age_model.backbone.print_trainable_parameters())

    n_total     = sum(p.numel() for p in age_model.parameters())
    n_trainable = sum(p.numel() for p in age_model.parameters() if p.requires_grad)
    print(f"Total params     : {n_total:,}")
    print(f"Trainable params : {n_trainable:,}  (LoRA Finetuning)\n")

    writer.add_text("model/architecture", str(age_model))
    writer.add_text("model/params", f"Total: {n_total:,} | Trainable: {n_trainable:,}")

    # ── Optimiser, scheduler, loss ───────────────────────────────────
    # Collected once, while PEFT still has them marked trainable.
    lora_params = [p for p in age_model.backbone.parameters() if p.requires_grad]

    def make_optimizer(stage):
        """Stage 1: head only, constant LR. Stage 2: head + LoRA, cosine over --epochs."""
        train_lora = stage == 2
        for p in lora_params:
            p.requires_grad_(train_lora)   # frozen -> AgePredictor runs the backbone under no_grad
        groups = [{"params": age_model.head.parameters(), "lr": args.lr_head}]
        if train_lora:
            groups.insert(0, {"params": lora_params, "lr": args.lr_lora})
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        if train_lora:
            warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=2)
            cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - 5, eta_min=1e-6)
            sch  = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[5])
        else:
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        return opt, sch

    stage = 1 if args.head_epochs > 0 else 2
    optimizer, scheduler = make_optimizer(stage)
    criterion  = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler()

    # ── Training loop ────────────────────────────────────────────────
    best_mae, best_epoch = float("inf"), 0
    total_epochs = args.head_epochs + args.epochs
    print(f"Training for {total_epochs} epochs "
          f"({args.head_epochs} head-only + {args.epochs} head + LoRA)...\n")

    for epoch in range(1, total_epochs + 1):

        if stage == 1 and epoch == args.head_epochs + 1:
            stage = 2
            optimizer, scheduler = make_optimizer(stage)
            print("\n── Stage 2: LoRA unfrozen ──\n")

        train_loss = train_one_epoch(
            age_model, train_loader, optimizer,
            criterion, scaler_amp, device, args.clip_grad,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        recalibrate_bn(age_model, train_loader, device)
        valid_mae, _, _, _ = evaluate(age_model, valid_loader, scaler, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("mae/valid",  valid_mae,  epoch)
        writer.add_scalar("lr",         current_lr, epoch)

        #if epoch % args.log_every == 0:
        #    log_attention_weights(age_model, valid_loader, pad_id, device, writer, epoch)

        if valid_mae < best_mae:
            best_mae, best_epoch = valid_mae, epoch
            torch.save(
                {
                    "epoch"          : epoch,
                    "run_id"         : run_id,
                    "model_state"    : age_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "mae"            : best_mae,
                    "run_config"     : run_config,
                },
                ckpt_dir / "best_model.pt",
            )

        print(
            f"[S{stage}] Epoch {epoch:3d}/{total_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Valid MAE: {valid_mae:.2f} yrs | "
            f"Best: {best_mae:.2f} yrs (ep {best_epoch}) | "
            f"LR: {current_lr:.2e}"
        )

    writer.add_scalar("mae/best_valid", best_mae, best_epoch)
    print(f"\nBest valid MAE: {best_mae:.2f} years at epoch {best_epoch}")

    # ── Test evaluation ──────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    checkpoint = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    age_model.load_state_dict(checkpoint["model_state"])

    test_mae, test_preds, test_labels, test_ids = evaluate(age_model, test_loader, scaler, device)
    test_r2   = r2_score(test_labels, test_preds)
    test_r, _ = pearsonr(test_labels, test_preds)

    print(f"\n{'─'*40}")
    print(f"  Test MAE      : {test_mae:.2f} years")
    print(f"  Test R²       : {test_r2:.3f}")
    print(f"  Test Pearson r: {test_r:.3f}")
    print(f"{'─'*40}\n")

    writer.add_scalar("mae/test",       test_mae, 0)
    writer.add_scalar("r2/test",        test_r2,  0)
    writer.add_scalar("pearson_r/test", test_r,   0)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(test_labels, test_preds, s=15, c="black",    alpha=1.0,  zorder=1)
        ax.scatter(test_labels, test_preds, s=10, alpha=0.5, c="steelblue", zorder=2)
        lims = [min(test_labels.min(), test_preds.min()),
                max(test_labels.max(), test_preds.max())]
        ax.plot(lims, lims, "r--", alpha=0.8, label="y = x")
        ax.set_xlabel("Chronological Age (years)")
        ax.set_ylabel("Predicted Age (years)")
        ax.set_title(f"{run_id[:40]}\nMAE={test_mae:.1f} yrs | r={test_r:.3f}")
        ax.legend(frameon=False)
        plt.tight_layout()
        writer.add_figure("test/scatter", fig)
        plt.savefig(log_path / "scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("matplotlib not available — skipping scatter plot")

    metrics = {
        "run_id"         : run_id,
        "best_epoch"     : best_epoch,
        "best_valid_mae" : best_mae,
        "test_mae"       : test_mae,
        "test_r2"        : test_r2,
        "test_pearson_r" : test_r,
        "test_ids"       : test_ids,
    }
    with open(log_path / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    writer.flush()
    writer.close()

    print(f"Logs        : {log_path}")
    print(f"Checkpoints : {ckpt_dir}")
    print(f"\nTo view all runs:\n  tensorboard --logdir {args.log_dir}\n")


if __name__ == "__main__":
    main()
