"""
EpiHydra: Core model classes for methylation-based age prediction.

Classes
-------
VocabWrapper          : Builds vocab from pretrained vocab.json
PreTokenizedDataset   : Cached tokenised dataset with masking collater
AttentionPoolingHead  : Cross-attention pooling over full CpG sequence
AgePredictor          : Frozen MethylGPT backbone + AttentionPoolingHead
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
        sorted_probes  = sorted(probe_to_id.items(), key=lambda x: x[1])
        self.CpG_list  = [probe for probe, _   in sorted_probes]
        token_ids      = [tid   for _,     tid in sorted_probes]

        self.CpG_ids     = np.array(token_ids, dtype=np.int64)
        self.vocab       = Vocab(VocabPybind(self.CpG_list, None))
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.max_seq_len = n_hvg + 1
        self.pad_token   = "<pad>"
        self.pad_value   = -2
        self.mask_value  = -1


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AgeDataset(torch.utils.data.Dataset):
    """Dataset for age prediction finetuning."""
    
    def __init__(self, vocab_obj, df, scaler, clock_age = 'age', mask_ratio=0.15, mask_seed=42):
        self.vocab = vocab_obj
        self.scaler = scaler
        self.mask_ratio = mask_ratio
        self.mask_seed = mask_seed
        self.gene_datas = df["data"].to_list()
        self.ages = df[clock_age].to_numpy()
        self.ages_norm = torch.tensor(
            scaler.transform(self.ages.reshape(-1, 1)), dtype=torch.float
        )
    
    def __getitem__(self, index):
        return self.gene_datas[index], torch.tensor(self.ages[index]).float(), self.ages_norm[index]
    
    def __len__(self):
        return len(self.ages)
    
    def collater(self, batch):
        gene_datas, ages, ages_norm = zip(*batch)
        
        # Process each sample individually with its own random mask
        all_gene_ids, all_masked, all_targets = [], [], []
        for sample in gene_datas:
            gene_ids, masked, target = self._tokenize(np.array([sample]))
            all_gene_ids.append(gene_ids)
            all_masked.append(masked)
            all_targets.append(target)
        
        return (
            torch.cat(all_gene_ids),
            torch.cat(all_masked),
            torch.cat(all_targets),
            torch.stack(ages),
            torch.stack(ages_norm)
        )

    def _tokenize(self, data):
        # Work entirely in numpy — no unnecessary tensor conversion
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data = np.nan_to_num(data, nan=-2.0)  # numpy equivalent of torch.nan_to_num
        
        tokenized = tokenize_and_pad_batch(
            data, self.vocab.CpG_ids,
            max_len=self.vocab.max_seq_len,
            vocab=self.vocab.vocab,
            pad_token="<pad>", pad_value=-2,
            append_cls=True, include_zero_gene=True,
        )
        masked = random_mask_value(
            tokenized["values"], mask_ratio=np.random.uniform(0.0, self.mask_ratio),
            mask_value=-1, pad_value=-2, mask_seed=np.random.randint(0, 1_000_000),
        )
        return tokenized["genes"], masked, tokenized["values"]

class MultiClockDataset(torch.utils.data.Dataset):
    """
    Dataset for training several epigenetic-clock heads at once off a single
    shared (frozen) backbone.

    Unlike AgeDataset, a sample may be missing the ground-truth label for
    some clocks (e.g. a GEO series that only reports Horvath age for a
    subset of samples). Missing labels are kept as NaN and paired with a
    per-clock validity mask so the training loop can zero out that clock's
    loss term for that sample rather than dropping the sample entirely.

    Parameters
    ----------
    vocab_obj  : VocabWrapper
    df         : DataFrame — must contain a "data" column (CpG array) and
                 one column per entry in `clock_names`
    scalers    : dict[str, MinMaxScaler] — one fitted scaler per clock,
                 fit only on that clock's non-NaN values
    clock_names: list[str]
    mask_ratio : float
    """

    def __init__(self, vocab_obj, df, scalers, clock_names, mask_ratio=0.15):
        self.vocab = vocab_obj
        self.scalers = scalers
        self.clock_names = list(clock_names)
        self.mask_ratio = mask_ratio
        self.gene_datas = df["data"].to_list()

        self.ages = {}
        self.valid = {}
        self.ages_norm = {}
        for name in self.clock_names:
            raw = df[name].to_numpy(dtype=np.float64)
            valid = ~np.isnan(raw)
            self.ages[name] = raw
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
        ages         = {n: float(self.ages[n][index])       for n in self.clock_names}
        valid        = {n: bool(self.valid[n][index])        for n in self.clock_names}
        ages_norm    = {n: float(self.ages_norm[n][index])   for n in self.clock_names}
        return self.gene_datas[index], ages, valid, ages_norm

    def collater(self, batch):
        gene_datas, ages, valid, ages_norm = zip(*batch)

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
        )

    def _tokenize(self, data):
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
        masked = random_mask_value(
            tokenized["values"], mask_ratio=np.random.uniform(0.0, self.mask_ratio),
            mask_value=-1, pad_value=-2, mask_seed=np.random.randint(0, 1_000_000),
        )
        return tokenized["genes"], masked, tokenized["values"]


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
# Head
# ──────────────────────────────────────────────────────────────────────────────

class AttentionPoolingHead(nn.Module):
    """
    Cross-attention pooling over the full CpG sequence.

    A single learned query vector attends over all non-padded CpG
    representations, producing a weighted summary projected to a
    scalar age prediction in [0, 1].

    Parameters
    ----------
    embed_dim : int   — must match backbone layer_size
    num_heads : int   — must divide embed_dim
    dropout   : float
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
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
        attn_weights : (batch, num_heads, 1, seq_len)
        """
        batch = sequence.shape[0]
        query = self.query.expand(batch, -1, -1)

        attended, attn_weights = self.cross_attn(
            query, sequence, sequence,
            key_padding_mask = key_padding_mask,
            need_weights     = True,
        )

        pooled = self.norm(attended.squeeze(1) + query.squeeze(1))
        pooled = self.dropout(pooled)

        return self.mlp(pooled).squeeze(-1), attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────


class AgePredictor(nn.Module):
    """
    Finetuned MethylGPT backbone with a trainable AttentionPoolingHead.

    Full-sequence hidden states are captured via a forward hook on
    transformer_encoder. A learned query vector then attends over all
    non-padded CpG representations to produce a scalar age prediction.

    Parameters
    ----------
    backbone  : TransformerModel — pretrained scgpt backbone
    embed_dim : int              — must match backbone layer_size
    pad_id    : int              — integer ID of <pad> token
    num_heads : int              — attention heads in pooling head
    dropout   : float
    """

    def __init__(self, backbone, embed_dim, pad_id, num_heads=4, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.pad_id   = pad_id
        self.head     = AttentionPoolingHead(embed_dim, num_heads, dropout)


    def forward(self, gene_ids, values):
        """
        Forward pass through frozen backbone and trainable attention head.

        Hidden states from the final transformer encoder layer are captured
        via a forward hook and passed to the attention pooling head. The hook
        is stored in a local dict rather than on self to avoid shared state
        across concurrent calls.

        Parameters
        ----------
        gene_ids : (batch, seq_len) int tensor  — tokenised CpG site IDs
        values   : (batch, seq_len) float tensor — masked methylation values

        Returns
        -------
        pred : (batch,) float tensor — predicted age in [0, 1]
        """
        src_key_padding_mask = gene_ids.eq(self.pad_id)

        # Capture full sequence hidden states locally — thread safe
        captured = {}
        hook = self.backbone.transformer_encoder.register_forward_hook(
            lambda module, inp, out: captured.update({"hidden": out})
        )

        # Only build an autograd graph through the backbone if it actually
        # has trainable params (e.g. LoRA adapters). When the backbone is
        # fully frozen (train_clock_heads.py), no_grad saves memory/compute.
        # NOTE: previously this was an unconditional `torch.no_grad()`, which
        # silently prevented LoRA weights from ever receiving gradients even
        # when they were marked requires_grad=True and included in the
        # optimizer — the graph was severed before backprop could reach them.
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        grad_ctx = torch.enable_grad() if backbone_trainable else torch.no_grad()

        with grad_ctx:
            _ = self.backbone(
                gene_ids, values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )

        hook.remove()

        sequence_hidden = captured["hidden"]

        pred, _ = self.head(
            sequence_hidden,
            key_padding_mask=src_key_padding_mask,
        )
        return pred


class MultiClockPredictor(nn.Module):
    """
    Frozen MethylGPT backbone + one independent AttentionPoolingHead per
    epigenetic clock.

    The backbone forward pass — the expensive part — is run exactly ONCE
    per batch. The resulting sequence hidden state is then fanned out
    through every clock head. Because the backbone is frozen (see `encode`,
    always under no_grad), each head trains as an independent linear probe
    on a shared, cached-in-memory representation: no cross-clock gradient
    interaction, and no repeated backbone forward passes per clock.

    Parameters
    ----------
    backbone    : TransformerModel — pretrained (optionally LoRA-merged)
                  backbone. Must already be frozen (requires_grad=False on
                  all params) — this class always runs it under no_grad.
    embed_dim   : int  — must match backbone layer_size
    pad_id      : int  — integer ID of <pad> token
    clock_names : list[str] — one head is created per name
    num_heads   : int  — attention heads inside each pooling head
    dropout     : float
    """

    def __init__(self, backbone, embed_dim, pad_id, clock_names, num_heads=4, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.pad_id = pad_id
        self.clock_names = list(clock_names)
        self.heads = nn.ModuleDict({
            name: AttentionPoolingHead(embed_dim, num_heads, dropout)
            for name in self.clock_names
        })

    def encode(self, gene_ids, values):
        """
        Single backbone forward pass, shared across all clock heads.

        Returns
        -------
        sequence_hidden : (batch, seq_len, embed_dim) — detached from graph
                           (backbone is frozen, always run under no_grad)
        key_padding_mask: (batch, seq_len) bool
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
        preds = {}
        for name in names:
            pred, _ = self.heads[name](sequence_hidden, key_padding_mask=mask)
            preds[name] = pred
        return preds