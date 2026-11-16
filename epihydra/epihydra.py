"""
EpiHydra: Core model classes for methylation-based age prediction.

Classes / Functions
-------------------
VocabWrapper          : Builds vocab from pretrained vocab.json
precompute_cpg_ranks  : Compute low/high-variance CpG rank arrays from training data
AgeDataset            : Single-clock dataset with stochastic mixed masking
MultiClockDataset     : Multi-clock dataset with stochastic mixed masking
load_backbone         : Instantiate and load pretrained TransformerModel
AttentionPoolingHead  : Cross-attention pooling over full CpG sequence
AgePredictor          : MethylGPT backbone + AttentionPoolingHead (LoRA finetuning)
MultiClockPredictor   : Frozen backbone + one AttentionPoolingHead per clock

Masking strategy
----------------
Each sample independently draws its masking strategy at collation time from a
user-defined probability distribution.  The default is 60 % uniform (fully
random), 20 % low-variance-first, and 20 % high-variance-first.  This means
every batch naturally contains all three regime types without any epoch-level
bookkeeping in the training loop.

The rank arrays are computed once from the *training* split and stored in a
dict keyed by strategy name.  Pass ``mask_weights={"uniform": 1.0}`` to
recover the original fully-random behaviour.
"""

import warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*nested tensor.*")

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from methylgpt import MethylGPTModel, MethylVocab
from scgpt.model.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

# ──────────────────────────────────────────────────────────────────────────────
# Vocab
# ──────────────────────────────────────────────────────────────────────────────

class VocabWrapper:
    """
    Builds a torchtext Vocab from a pretrained MethylGPT vocab JSON.

    The JSON maps probe IDs to their pretrained token IDs:
        {"cg00000029": 3, "ch.9.1059422R": 49158, ...}

    Probes are sorted by token ID to reconstruct the exact probe order
    used during pretraining.

    Parameters
    ----------
    vocab_json_file : str | Path
    n_hvg           : int — number of CpG probes the backbone expects
    """

    def __init__(self, vocab_json_file, n_hvg):
        from torchtext.vocab import Vocab
        from torchtext._torchtext import Vocab as VocabPybind

        with open(vocab_json_file, "r") as f:
            probe_to_id = json.load(f)

        # Sort by token ID — reconstructs exact pretraining probe order
        sorted_probes = sorted(probe_to_id.items(), key=lambda x: x[1])
        self.CpG_list = [probe for probe, _   in sorted_probes]
        token_ids     = [tid   for _,     tid in sorted_probes]

        self.CpG_ids     = np.array(token_ids, dtype=np.int64)
        self.vocab       = Vocab(VocabPybind(self.CpG_list, None))
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.max_seq_len = n_hvg + 1
        self.pad_token   = "<pad>"
        self.pad_value   = -2
        self.mask_value  = -1


# ──────────────────────────────────────────────────────────────────────────────
# CpG variance ranking
# ──────────────────────────────────────────────────────────────────────────────

MASK_MODES = ("uniform", "low_variance", "high_variance")

# Default per-sample strategy probabilities used by both dataset classes.
# Override via mask_weights= at construction time or from the CLI.
DEFAULT_MASK_WEIGHTS = {"uniform": 0.9, "low_variance": 0.0, "high_variance": 0.1}


def precompute_cpg_ranks(train_df):
    """
    Compute CpG rank arrays for all three masking strategies in one variance
    pass over the training split.

    Call this once after the train/val/test split, then pass the result as
    ``all_cpg_ranks`` to AgeDataset / MultiClockDataset.  Using only the
    training split avoids leaking validation or test statistics into the mask.

    Parameters
    ----------
    train_df : DataFrame
        Must contain a "data" column of CpG value arrays
        (shape n_samples x n_cpgs, may contain NaN).

    Returns
    -------
    ranks : dict[str, np.ndarray]
        "uniform"       -> np.arange(n_cpgs)  (natural vocab order)
        "low_variance"  -> ascending by per-probe variance (least variable first)
        "high_variance" -> descending by per-probe variance (most variable first)

        NaN probes (always missing) are sorted last in both ranked orderings so
        they never consume mask budget that could target real signal.
    """
    n_cpgs = len(train_df["data"].iloc[0])
    X      = np.stack(train_df["data"].to_list())   # (n_samples, n_cpgs)

    with np.errstate(all="ignore"):
        var = np.nanvar(X, axis=0)                  # (n_cpgs,) — NaN where all-NaN probe

    # Push NaN-variance probes to the end for both orderings
    low_stat  = np.where(np.isfinite(var), var,  np.inf)
    high_stat = np.where(np.isfinite(var), var, -np.inf)

    return {
        "uniform"      : np.arange(n_cpgs, dtype=int),
        "low_variance" : np.argsort(low_stat).astype(int),
        "high_variance": np.argsort(high_stat)[::-1].astype(int),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shared masking mixin
# ──────────────────────────────────────────────────────────────────────────────

class _MaskingMixin:
    """
    Shared masking logic for AgeDataset and MultiClockDataset.

    Subclasses must set:
        self.mask_ratio    : float
        self.all_cpg_ranks : dict[str, np.ndarray]
        self._mask_modes   : list[str]
        self._mask_probs   : np.ndarray (normalised probabilities)
    """

    def _sample_mode(self):
        """Draw a masking strategy for one sample according to mask_weights."""
        return np.random.choice(self._mask_modes, p=self._mask_probs)

    def _apply_mask(self, tokenized):
        """
        Apply the sampled masking strategy to a tokenized batch of one sample.

        Parameters
        ----------
        tokenized : dict — output of tokenize_and_pad_batch, with keys
                    "genes" and "values".

        Returns
        -------
        genes  : token-id tensor  (1, seq_len)
        masked : value tensor     (1, seq_len) with mask_value=-1 applied
        values : original value tensor (1, seq_len), unmasked targets
        """
        mode = self._sample_mode()

        if mode == "uniform":
            # Fully random masking — identical to the original behaviour
            masked = random_mask_value(
                tokenized["values"],
                mask_ratio=np.random.uniform(0.0, self.mask_ratio),
                mask_value=-1,
                pad_value=-2,
                mask_seed=np.random.randint(0, 1_000_000),
            )
        else:
            masked = self._ranked_mask(tokenized["values"], mode)

        return tokenized["genes"], masked, tokenized["values"]

    def _ranked_mask(self, values, mode):
        """
        Mask the top-ranked CpGs for the given variance strategy.

        The CLS token (position 0) is never masked.  Pad slots (value == -2)
        are skipped so the mask budget is spent on real CpG signal only.

        Parameters
        ----------
        values : torch.Tensor, shape (1, seq_len)
        mode   : "low_variance" | "high_variance"

        Returns
        -------
        masked : torch.Tensor, shape (1, seq_len)
        """
        masked  = values.clone()
        seq     = masked[0]                         # (seq_len,)
        n_total = seq.shape[0] - 1                  # exclude CLS at position 0

        n_mask = int(np.random.uniform(0.0, self.mask_ratio) * n_total)
        if n_mask == 0:
            return masked

        # Vocab position i sits at sequence position i+1 (CLS occupies pos 0)
        masked_count = 0
        for vocab_pos in self.all_cpg_ranks[mode]:
            if masked_count >= n_mask:
                break
            seq_pos = int(vocab_pos) + 1
            if seq_pos >= seq.shape[0]:
                continue
            if seq[seq_pos].item() == -2:           # pad — skip
                continue
            seq[seq_pos] = -1
            masked_count += 1

        return masked

    def _tokenize(self, data):
        """Tokenize a single-sample array and apply mixed masking."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data = np.nan_to_num(data, nan=-2.0)

        tokenized = tokenize_and_pad_batch(
            data, self.vocab.CpG_ids,
            max_len=self.vocab.max_seq_len,
            vocab=self.vocab.vocab,
            pad_token="<pad>", pad_value=-2,
            append_cls=True, include_zero_gene=True,
        )
        return self._apply_mask(tokenized)

    @staticmethod
    def _validate_weights(weights, all_cpg_ranks):
        """Raise if weights reference unknown modes or ranks are missing."""
        for k in weights:
            if k not in MASK_MODES:
                raise ValueError(
                    f"Unknown mask mode '{k}'; must be one of {MASK_MODES}."
                )
        needs_ranks = any(w > 0 for k, w in weights.items() if k != "uniform")
        if needs_ranks and not all_cpg_ranks:
            raise ValueError(
                "all_cpg_ranks must be provided when any ranked mode has "
                "non-zero weight.  Call precompute_cpg_ranks(train_df) first."
            )

    @staticmethod
    def _normalise_weights(weights):
        """Return parallel (modes, probs) lists normalised to sum to 1."""
        total = sum(weights.values())
        modes = list(weights.keys())
        probs = np.array([weights[m] / total for m in modes])
        return modes, probs


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class AgeDataset(_MaskingMixin, torch.utils.data.Dataset):
    """
    Dataset for single-clock age prediction finetuning.

    Each sample independently draws its masking strategy at collation time
    from ``mask_weights``, so every batch contains a natural mix of masking
    regimes with no epoch-level bookkeeping in the training loop.

    Parameters
    ----------
    vocab_obj     : VocabWrapper
    df            : DataFrame — must contain "data" and "age" columns.
                    The DataFrame index is used as sample IDs.
    scaler        : fitted MinMaxScaler for age normalisation.
    mask_ratio    : float
        Maximum fraction of tokens to mask per sample.  The actual fraction
        is sampled uniformly in [0, mask_ratio] for every sample.
    mask_seed     : int — kept for API compatibility; seeds are drawn randomly.
    mask_weights  : dict[str, float] | None
        Sampling probabilities for each strategy, e.g.
        ``{"uniform": 0.6, "low_variance": 0.2, "high_variance": 0.2}``.
        Values are normalised to sum to 1.  Defaults to DEFAULT_MASK_WEIGHTS.
        Pass ``{"uniform": 1.0}`` to recover the original fully-random
        behaviour with no variance-ranked masking.
    all_cpg_ranks : dict[str, np.ndarray] | None
        Pre-computed output of ``precompute_cpg_ranks(train_df)``.  Required
        when any ranked mode carries non-zero weight.
    """

    def __init__(
        self,
        vocab_obj,
        df,
        scaler,
        clock_age='age',
        mask_ratio=0.15,
        mask_seed=42,
        mask_weights=None,
        all_cpg_ranks=None,
    ):
        weights = mask_weights or DEFAULT_MASK_WEIGHTS
        self._validate_weights(weights, all_cpg_ranks)

        self.vocab         = vocab_obj
        self.scaler        = scaler
        self.mask_ratio    = mask_ratio
        self.all_cpg_ranks = all_cpg_ranks or {}
        self.gene_datas    = df["data"].to_list()
        self.ages          = df[clock_age].to_numpy()
        self.sample_ids    = list(df.index)
        self.ages_norm     = torch.tensor(
            scaler.transform(self.ages.reshape(-1, 1)), dtype=torch.float
        )
        self._mask_modes, self._mask_probs = self._normalise_weights(weights)

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, index):
        return (
            self.gene_datas[index],
            torch.tensor(self.ages[index]).float(),
            self.ages_norm[index],
            self.sample_ids[index],
        )

    def collater(self, batch):
        gene_datas, ages, ages_norm, ids = zip(*batch)

        all_gene_ids, all_masked, all_targets = [], [], []
        for sample in gene_datas:
            gene_ids, masked, target = self._tokenize(np.array([sample]))
            all_gene_ids.append(gene_ids)
            all_masked.append(masked)
            all_targets.append(target)

        return (
            torch.cat(all_gene_ids),   # gene_ids
            torch.cat(all_masked),     # masked_vals
            torch.cat(all_targets),    # target_vals (unmasked, for eval)
            torch.stack(ages),         # ages in years
            torch.stack(ages_norm),    # ages normalised
            list(ids),                 # sample IDs
        )


class MultiClockDataset(_MaskingMixin, torch.utils.data.Dataset):
    """
    Dataset for training several epigenetic-clock heads simultaneously off a
    single shared backbone.

    A sample may be missing the ground-truth label for some clocks (e.g. a GEO
    series that only reports Horvath age for a subset of samples).  Missing
    labels are kept as NaN and paired with a per-clock validity mask so the
    training loop can zero out that clock's loss term for that sample rather
    than dropping the sample entirely.

    Masking follows the same stochastic per-sample strategy as AgeDataset.
    See that class for full mask_weights / all_cpg_ranks documentation.

    Parameters
    ----------
    vocab_obj     : VocabWrapper
    df            : DataFrame — must contain a "data" column plus one column
                    per entry in ``clock_names``.
    scalers       : dict[str, MinMaxScaler] — one fitted scaler per clock,
                    fit only on that clock's non-NaN training values.
    clock_names   : list[str]
    mask_ratio    : float
    mask_weights  : dict[str, float] | None — defaults to DEFAULT_MASK_WEIGHTS.
    all_cpg_ranks : dict[str, np.ndarray] | None
    """

    def __init__(
        self,
        vocab_obj,
        df,
        scalers,
        clock_names,
        mask_ratio=0.15,
        mask_weights=None,
        all_cpg_ranks=None,
    ):
        weights = mask_weights or DEFAULT_MASK_WEIGHTS
        self._validate_weights(weights, all_cpg_ranks)

        self.vocab         = vocab_obj
        self.scalers       = scalers
        self.clock_names   = list(clock_names)
        self.mask_ratio    = mask_ratio
        self.all_cpg_ranks = all_cpg_ranks or {}
        self.gene_datas    = df["data"].to_list()
        self._mask_modes, self._mask_probs = self._normalise_weights(weights)
        self.sample_ids    = list(df.index)

        self.ages      = {}
        self.valid     = {}
        self.ages_norm = {}
        for name in self.clock_names:
            raw   = df[name].to_numpy(dtype=np.float64)
            valid = ~np.isnan(raw)
            self.ages[name]  = raw
            self.valid[name] = valid

            norm = np.zeros_like(raw, dtype=np.float32)
            if valid.any():
                norm[valid] = self.scalers[name].transform(
                    raw[valid].reshape(-1, 1)
                ).flatten()
            self.ages_norm[name] = norm

    def __len__(self):
        return len(self.gene_datas)

    def __getitem__(self, index):
        ages      = {n: float(self.ages[n][index])      for n in self.clock_names}
        valid     = {n: bool(self.valid[n][index])      for n in self.clock_names}
        ages_norm = {n: float(self.ages_norm[n][index]) for n in self.clock_names}
        return self.gene_datas[index], ages, valid, ages_norm, str(self.sample_ids[index])

    def collater(self, batch):
        gene_datas, ages, valid, ages_norm, ids = zip(*batch)

        all_gene_ids, all_masked, all_targets = [], [], []
        for sample in gene_datas:
            gene_ids, masked, target = self._tokenize(np.array([sample]))
            all_gene_ids.append(gene_ids)
            all_masked.append(masked)
            all_targets.append(target)

        ages_out, valid_out, norm_out = {}, {}, {}
        for name in self.clock_names:
            ages_out[name]  = torch.tensor([a[name] for a in ages],  dtype=torch.float)
            valid_out[name] = torch.tensor([v[name] for v in valid], dtype=torch.bool)
            norm_out[name]  = torch.tensor([n[name] for n in ages_norm], dtype=torch.float)

        return (
            torch.cat(all_gene_ids),
            torch.cat(all_masked),
            torch.cat(all_targets),
            ages_out,
            valid_out,
            norm_out,
            list(ids),                 # sample IDs
        )


# ──────────────────────────────────────────────────────────────────────────────
# Backbone loader
# ──────────────────────────────────────────────────────────────────────────────

def load_backbone(model_file, vocab, pretrain_config):
    """
    Instantiate TransformerModel and load pretrained weights.
    Falls back to partial loading if the checkpoint has extra keys.
    """
    backbone = TransformerModel(
        len(vocab.vocab),
        pretrain_config["layer_size"],
        pretrain_config["nhead"],
        pretrain_config["layer_size"],
        pretrain_config["nlayers"],
        vocab                 = vocab.vocab,
        dropout               = pretrain_config.get("dropout", 0.0),
        pad_token             = "<pad>",
        pad_value             = -2,
        do_mvc                = True,
        do_dab                = False,
        use_batch_labels      = False,
        num_batch_labels      = None,
        domain_spec_batchnorm = False,
        n_input_bins          = None,
        ecs_threshold         = 0.0,
        explicit_zero_prob    = False,
        use_fast_transformer  = pretrain_config.get("fast_transformer", False),
        pre_norm              = pretrain_config.get("pre_norm", False),
    )

    pretrained = torch.load(model_file, map_location="cpu")
    try:
        backbone.load_state_dict(pretrained)
        print(f"Loaded all params from {model_file}")
    except RuntimeError:
        model_dict = backbone.state_dict()
        compatible = {
            k: v for k, v in pretrained.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(compatible)
        backbone.load_state_dict(model_dict)
        print(f"Loaded {len(compatible)}/{len(pretrained)} compatible params")

    return backbone


# ──────────────────────────────────────────────────────────────────────────────
# Pooling head
# ──────────────────────────────────────────────────────────────────────────────

class AttentionPoolingHead(nn.Module):
    """
    Multi-query cross-attention pooling over the full CpG sequence.

    Multiple learned query vectors attend over all non-padded CpG
    representations.  Their outputs are mean-pooled and passed through
    a deeper MLP to produce a scalar age prediction in [0, 1].

    Using n_queries > 1 allows the head to simultaneously attend to different
    biologically relevant CpG regions (e.g. one query may specialise on
    tissue-specific sites, another on housekeeping loci).

    Parameters
    ----------
    embed_dim : int   — must match backbone layer_size
    num_heads : int   — must divide embed_dim
    dropout   : float
    n_queries : int   — number of learned query vectors (default 4)
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1, n_queries=4):
        super().__init__()
        
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm    = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Deeper MLP: embed_dim -> embed_dim -> embed_dim//2 -> 1
        self.mlp = nn.Sequential(
#            nn.Linear(embed_dim, embed_dim),
#            nn.GELU(),
#            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
#            nn.Sigmoid(),
        )

    def forward(self, sequence, key_padding_mask=None):
        """
        Parameters
        ----------
        sequence         : (batch, seq_len, embed_dim)
        key_padding_mask : (batch, seq_len) bool — True = ignore

        Returns
        -------
        pred         : (batch,)
        attn_weights : (batch, num_heads, n_queries, seq_len)
        """
        batch = sequence.shape[0]
        query = self.query.expand(batch, -1, -1)

        attended, attn_weights = self.cross_attn(
            query, sequence, sequence,
            key_padding_mask = key_padding_mask,
            need_weights     = True,
        )

        pooled = self.norm(attended.squeeze(1))
        pooled = self.dropout(pooled)

        return self.mlp(pooled).squeeze(-1), attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# Full models
# ──────────────────────────────────────────────────────────────────────────────

class AgePredictor(nn.Module):
    """
    Finetuned MethylGPT backbone + trainable AttentionPoolingHead.

    Full-sequence hidden states are captured via a forward hook on
    transformer_encoder.  A learned query vector then attends over all
    non-padded CpG representations to produce a scalar age prediction.

    The backbone is run under torch.enable_grad() when it carries trainable
    parameters (e.g. LoRA adapters), and under torch.no_grad() otherwise.

    Parameters
    ----------
    backbone  : TransformerModel — pretrained scgpt backbone
    embed_dim : int              — must match backbone layer_size
    pad_id    : int              — integer ID of <pad> token
    num_heads : int              — attention heads in pooling head
    dropout   : float
    n_queries : int              — number of learned query vectors
    """

    def __init__(self, backbone, embed_dim, pad_id, num_heads=4, dropout=0.1, n_queries=4):
        super().__init__()
        self.backbone = backbone
        self.pad_id   = pad_id
        self.head     = AttentionPoolingHead(embed_dim, num_heads, dropout, n_queries)

    def forward(self, gene_ids, values):
        """
        Parameters
        ----------
        gene_ids : (batch, seq_len) int tensor  — tokenised CpG site IDs
        values   : (batch, seq_len) float tensor — masked methylation values

        Returns
        -------
        pred : (batch,) float tensor — predicted age in [0, 1]
        """
        src_key_padding_mask = gene_ids.eq(self.pad_id)

        captured = {}
        hook = self.backbone.transformer_encoder.register_forward_hook(
            lambda module, inp, out: captured.update({"hidden": out})
        )

        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        grad_ctx = torch.enable_grad() if backbone_trainable else torch.no_grad()

        with grad_ctx:
            _ = self.backbone(
                gene_ids, values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )

        hook.remove()

        pred, _ = self.head(
            captured["hidden"],
            key_padding_mask=src_key_padding_mask,
        )
        return pred


class MultiClockPredictor(nn.Module):
    """
    Frozen MethylGPT backbone + one independent AttentionPoolingHead per clock.

    The backbone forward pass is run exactly once per batch under no_grad.
    The resulting hidden state is fanned out to every clock head, so each head
    trains as an independent linear probe on a shared representation with no
    cross-clock gradient interaction.

    Parameters
    ----------
    backbone    : TransformerModel — pretrained, fully frozen backbone.
    embed_dim   : int
    pad_id      : int
    clock_names : list[str] — one head per name
    num_heads   : int
    dropout     : float
    n_queries   : int
    """

    def __init__(
        self,
        backbone,
        embed_dim,
        pad_id,
        clock_names,
        num_heads=4,
        dropout=0.1,
        n_queries=4,
    ):
        super().__init__()
        self.backbone    = backbone
        self.pad_id      = pad_id
        self.clock_names = list(clock_names)
        self.heads       = nn.ModuleDict({
            name: AttentionPoolingHead(embed_dim, num_heads, dropout, n_queries)
            for name in self.clock_names
        })

    def encode(self, gene_ids, values):
        """
        Single backbone forward pass shared across all clock heads.

        Returns
        -------
        sequence_hidden  : (batch, seq_len, embed_dim)
        key_padding_mask : (batch, seq_len) bool
        """
        src_key_padding_mask = gene_ids.eq(self.pad_id)

        captured = {}
        hook = self.backbone.transformer_encoder.register_forward_hook(
            lambda module, inp, out: captured.update({"hidden": out})
        )
        with torch.no_grad():
            _ = self.backbone(
                gene_ids, values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )
        hook.remove()

        return captured["hidden"], src_key_padding_mask

    def forward(self, gene_ids, values, clock_names=None):
        """
        Returns
        -------
        dict[str, (batch,) tensor] — predictions in [0, 1] per clock
        """
        sequence_hidden, mask = self.encode(gene_ids, values)
        names = clock_names if clock_names is not None else self.clock_names
        return {
            name: self.heads[name](sequence_hidden, key_padding_mask=mask)[0]
            for name in names
        }

# ──────────────────────────────────────────────────────────────────────────────
# Ridge Head
# ──────────────────────────────────────────────────────────────────────────────

class LinearProbeHead(nn.Module):
    """Mean over non-pad positions -> BatchNorm (per-feature standardisation) -> Linear."""
    def __init__(self, embed_dim):
        super().__init__()
        self.bn  = nn.BatchNorm1d(embed_dim)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, sequence, key_padding_mask=None):
        keep   = (~key_padding_mask).unsqueeze(-1).to(sequence.dtype)
        pooled = (sequence * keep).sum(1, dtype=torch.float32) / keep.sum(1, dtype=torch.float32).clamp(min=1)
        return self.out(self.bn(pooled)).squeeze(-1), None


@torch.no_grad()
def init_head_from_ridge(model, loader, age_scaler, device, pad_id):
    """Fit ridge on frozen mean-pooled features and copy it into a LinearProbeHead."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model.eval()
    feats, ages = [], []
    for gene_ids, _, target_vals, a, *_ in loader:
        gene_ids, vals = gene_ids.to(device), target_vals.to(device)
        mask, cap = gene_ids.eq(pad_id), {}
        hook = model.backbone.transformer_encoder.register_forward_hook(lambda m, i, o: cap.update(h=o))
        with torch.cuda.amp.autocast():                  # fp16, same as training and evaluate
            model.backbone(gene_ids, vals, src_key_padding_mask=mask, batch_labels=None)
        hook.remove()
        keep = (~mask).unsqueeze(-1).float()
        feats.append(((cap["h"].float() * keep).sum(1) / keep.sum(1)).cpu())
        ages.extend(a.numpy())

    probe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    probe.fit(torch.cat(feats).numpy(), np.array(ages))
    std, ridge = probe[0], probe[1]

    # Ridge predicts years; the model predicts min-max-scaled age, so rescale weights and bias.
    lo, span = age_scaler.data_min_[0], age_scaler.data_range_[0]
    head = model.head
    head.bn.running_mean.copy_(torch.as_tensor(std.mean_))
    head.bn.running_var.copy_(torch.as_tensor(std.var_))
    head.bn.weight.fill_(1.0)
    head.bn.bias.zero_()
    head.out.weight.copy_(torch.as_tensor(ridge.coef_ / span).view(1, -1))
    head.out.bias.fill_(float((ridge.intercept_ - lo) / span))