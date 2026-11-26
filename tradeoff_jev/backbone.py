"""Pretrained Qwen Transformer Backbone and Differentiable Neural Systems for Trade-Off Jev.

This module bridges compact Apache-2.0 pretrained transformer backbones (such as
`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen2.5-0.5B`, and `Qwen/Qwen3.5-0.8B`)
with single-forward-pass SystemOne / Jev decision heads:

1. `DecisionSequencePacker`: Packs `(state_text, candidate_descriptions)` into a single
   contiguous token sequence with exact token-span tracking (`[start, end)`) for every
   candidate $u_k \\in \\mathcal{X}_t$ and a terminal `[DECIDE]` query anchor.
2. `QwenDecisionBackbone`: Wraps the pretrained Qwen base transformer in `bfloat16`,
   explicitly strips the 151k–248k `lm_head` vocabulary projection so zero VRAM is spent
   on autoregressive token logits, attaches `peft` LoRA adapters (`r=16, lora_alpha=32`)
   onto attention and MLP projections, and pools candidate/decision representations in a
   single forward pass.
3. `TradeOffJevNeuralSystem`: Connects `QwenDecisionBackbone` to FP32 PyTorch autograd
   implementations of `TorchPreferenceTableEmbedding`, `TorchDifferentiableAATModule`,
   and `TorchConditionedNoulHead`, computing calibrated `tradeoff_noul`
   $P(\\text{yes} \\mid x_t, u_k, \\mathcal{T}_t)$, `bartleby_noul`
   $P(\\text{Bartleby} \\mid x_t, \\mathcal{T}_t)$, ordinal rubric logits, and
   differentiable Markov updates $\\mathcal{T}_{t-1} \\to \\hat{\\mathcal{T}}_t$.
4. `StandardJevBaselineSystem`: Controlled ablation sharing the exact same
   `QwenDecisionBackbone` and LoRA configuration, but replacing the AAT head with an
   unconditioned Jev pointer/noul head (`UnconditionedJevReadoutHead`) that receives
   neither `PreferenceTable` nor `Bartleby` survival-floor inputs.

References:
    - Selten, R. (1998). Aspiration Adaptation Theory. Journal of Mathematical Psychology.
    - Liu et al. (2026). Decision 1.0: Single-Pass Structured Readout Heads (arXiv:2611.18629).
    - Wiki: `wiki/sources/decision_models_2026.md`, `wiki/concepts/conditioned_nouls.md`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .model import AATModelConfig

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    HAS_PEFT = True
except ImportError:  # pragma: no cover
    HAS_PEFT = False


DEFAULT_QWEN_CHECKPOINT: str = "Qwen/Qwen3-0.6B"

DEFAULT_LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class PoolingStrategy(str, Enum):
    """Token-span pooling strategy for candidate latent extraction in single-pass heads.

    Attributes:
        SPAN_END: Extracts the final token representation (`end_idx - 1`) of each
            candidate span, leveraging causal attention accumulation across the state
            prefix and candidate description (Kev / Clef causal anchor style).
        SPAN_MEAN: Mean-pools token hidden states across `[start_idx, end_idx)` for
            each candidate span.
        SPAN_END_AND_MEAN: Averages the causal end-token state and mean-pooled span
            state to capture both boundary summary and distributed token features.
    """

    SPAN_END = "span_end"
    SPAN_MEAN = "span_mean"
    SPAN_END_AND_MEAN = "span_end_and_mean"


@dataclass(frozen=True)
class BackboneConfig:
    """Immutable configuration for `QwenDecisionBackbone` and LoRA adapter injection.

    Attributes:
        model_name_or_path: HuggingFace Hub ID or local directory path for the Qwen
            pretrained checkpoint (defaults to Apache-2.0 `Qwen/Qwen3-0.6B`).
        torch_dtype: Backbone floating-point precision (`"bfloat16"`, `"float16"`, or `"float32"`).
        max_seq_length: Maximum packed token sequence length $S_{\\max}$ per decision example.
        pooling_strategy: Span pooling strategy (`span_end`, `span_mean`, or `span_end_and_mean`).
        use_lora: Whether to freeze base weights and attach `peft` LoRA adapters.
        lora_r: Rank $r$ of low-rank adaptation matrices ($r = 16$).
        lora_alpha: Scaling numerator $\\alpha$ for LoRA updates ($\\alpha = 32$).
        lora_dropout: Dropout probability applied to LoRA input activations.
        target_modules: Candidate linear projection suffixes to adapt with LoRA.
        gradient_checkpointing: Whether to enable activation checkpointing on transformer
            layers to reduce peak VRAM on 8 GB GPUs.
        trust_remote_code: Passed to HuggingFace `AutoConfig` / `AutoModel` loaders.
    """

    model_name_or_path: str = DEFAULT_QWEN_CHECKPOINT
    torch_dtype: str = "bfloat16"
    max_seq_length: int = 1024
    pooling_strategy: PoolingStrategy = PoolingStrategy.SPAN_END_AND_MEAN
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        """Validate backbone and LoRA configuration invariants."""
        if not self.model_name_or_path or not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string.")
        if self.torch_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(
                f"Unsupported torch_dtype {self.torch_dtype!r}; expected 'bfloat16', 'float16', or 'float32'."
            )
        if self.max_seq_length < 32:
            raise ValueError(f"max_seq_length must be >= 32, got {self.max_seq_length}.")
        if self.use_lora and (self.lora_r < 1 or self.lora_alpha < 1):
            raise ValueError("lora_r and lora_alpha must be >= 1 when use_lora=True.")

    def resolved_torch_dtype(self) -> torch.dtype:
        """Return the `torch.dtype` corresponding to `self.torch_dtype`."""
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self.torch_dtype]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        raw = asdict(self)
        raw["pooling_strategy"] = (
            self.pooling_strategy.value
            if isinstance(self.pooling_strategy, PoolingStrategy)
            else str(self.pooling_strategy)
        )
        raw["target_modules"] = list(self.target_modules)
        return raw

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackboneConfig":
        """Reconstruct `BackboneConfig` from a serialized dictionary."""
        return cls(
            model_name_or_path=str(data.get("model_name_or_path", DEFAULT_QWEN_CHECKPOINT)),
            torch_dtype=str(data.get("torch_dtype", "bfloat16")),
            max_seq_length=int(data.get("max_seq_length", 1024)),
            pooling_strategy=PoolingStrategy(
                data.get("pooling_strategy", PoolingStrategy.SPAN_END_AND_MEAN.value)
            ),
            use_lora=bool(data.get("use_lora", True)),
            lora_r=int(data.get("lora_r", 16)),
            lora_alpha=int(data.get("lora_alpha", 32)),
            lora_dropout=float(data.get("lora_dropout", 0.05)),
            target_modules=tuple(data.get("target_modules", DEFAULT_LORA_TARGET_MODULES)),
            gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
        )


@dataclass(frozen=True)
class PackedDecisionBatch:
    """Immutable tensor batch representing packed `(state_text, candidates)` sequences.

    Attributes:
        input_ids: Token IDs of shape `[B, S]`, dtype `torch.long`.
        attention_mask: Attention mask of shape `[B, S]`, dtype `torch.long` (`1` = valid, `0` = pad).
        candidate_spans: Candidate token boundaries `[start_idx, end_idx)` of shape `[B, K, 2]`.
        decide_positions: Index of the final `[DECIDE]` anchor token of shape `[B]`.
        candidate_mask: Boolean validity mask over candidate slots of shape `[B, K]`.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    candidate_spans: torch.Tensor
    decide_positions: torch.Tensor
    candidate_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "PackedDecisionBatch":
        """Transfer all batch tensors onto the target PyTorch device."""
        return PackedDecisionBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            candidate_spans=self.candidate_spans.to(device, non_blocking=True),
            decide_positions=self.decide_positions.to(device, non_blocking=True),
            candidate_mask=self.candidate_mask.to(device, non_blocking=True),
        )


class DecisionSequencePacker:
    """Packs decision context $x_t$ and candidate descriptions $\\{u_1, \\dots, u_K\\}$ with span indices.

    Constructs a single structured sequence per decision step:
        `[STATE] <state_text> \\n [CANDIDATE 0] <cand_0> [/CANDIDATE] ... \\n [DECIDE]`

    By encoding each segment deterministically without implicit mid-sequence BOS/EOS
    insertions, `DecisionSequencePacker` records exact `[start_tok, end_tok)` spans for
    each candidate $u_k$ and guarantees that all candidate spans and the `[DECIDE]`
    anchor survive truncation when `state_text` is long.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int = 1024,
    ) -> None:
        if max_seq_length < 32:
            raise ValueError(f"max_seq_length must be >= 32, got {max_seq_length}.")
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad_token_id: int = self._resolve_pad_token_id(tokenizer)

    @staticmethod
    def _resolve_pad_token_id(tokenizer: Any) -> int:
        """Ensure the tokenizer exposes a valid integer `pad_token_id`."""
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, list) and eos_id:
            eos_id = eos_id[0]
        if eos_id is not None:
            if hasattr(tokenizer, "pad_token") and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return int(eos_id)
        return 0

    @property
    def pad_token_id(self) -> int:
        """Return the integer padding token ID."""
        return self._pad_token_id

    @property
    def max_seq_length(self) -> int:
        """Return the maximum packed sequence length."""
        return self._max_seq_length

    def _encode_segment(self, text: str) -> List[int]:
        """Encode a text segment without adding special BOS/EOS tokens."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [int(t) for t in ids] if ids else [self._pad_token_id]

    def pack_batch(
        self,
        state_texts: Sequence[str],
        candidate_descriptions: Sequence[Sequence[str]],
        device: Optional[torch.device | str] = None,
    ) -> PackedDecisionBatch:
        """Pack a batch of `(state_text, candidates)` pairs into a `PackedDecisionBatch`.

        Args:
            state_texts: Sequence of $B$ environment/context descriptions $x_t$.
            candidate_descriptions: Sequence of $B$ candidate string lists, where item $b$
                has $K_b \\ge 1$ candidate descriptions. Variable $K_b$ across a batch is
                automatically padded to $K_{\\max} = \\max_b K_b$ with `candidate_mask`.
            device: Optional target `torch.device` for returned tensors.

        Returns:
            `PackedDecisionBatch` containing `input_ids` `[B, S]`, `attention_mask` `[B, S]`,
            `candidate_spans` `[B, K_max, 2]`, `decide_positions` `[B]`, and
            `candidate_mask` `[B, K_max]`.
        """
        if len(state_texts) == 0:
            raise ValueError("state_texts must contain at least one example.")
        if len(state_texts) != len(candidate_descriptions):
            raise ValueError(
                f"Batch size mismatch: {len(state_texts)} state_texts vs "
                f"{len(candidate_descriptions)} candidate lists."
            )

        batch_size = len(state_texts)
        max_candidates = max(len(cands) for cands in candidate_descriptions)
        if max_candidates < 1:
            raise ValueError("Each example must provide at least one candidate description.")

        decide_tokens = self._encode_segment("\n[DECIDE]")
        all_token_ids: List[List[int]] = []
        all_spans: List[List[Tuple[int, int]]] = []
        all_decide_pos: List[int] = []
        all_cand_valid: List[List[bool]] = []

        for state_str, cands in zip(state_texts, candidate_descriptions):
            if len(cands) < 1:
                raise ValueError("Candidate list for an example cannot be empty.")

            # Encode candidate blocks first so we know exact token budget required after state
            cand_token_blocks: List[List[int]] = []
            for idx, cand_text in enumerate(cands):
                formatted = f"\n[CANDIDATE {idx}] {cand_text} [/CANDIDATE]"
                cand_token_blocks.append(self._encode_segment(formatted))

            suffix_budget = sum(len(blk) for blk in cand_token_blocks) + len(decide_tokens)
            # If candidates alone exceed max_seq_length, proportionally trim candidate blocks
            if suffix_budget >= self._max_seq_length - 4:
                per_cand_cap = max(
                    4,
                    (self._max_seq_length - len(decide_tokens) - 8) // max(1, len(cand_token_blocks)),
                )
                cand_token_blocks = [blk[:per_cand_cap] for blk in cand_token_blocks]
                suffix_budget = sum(len(blk) for blk in cand_token_blocks) + len(decide_tokens)

            state_budget = max(4, self._max_seq_length - suffix_budget)
            state_tokens = self._encode_segment(f"[STATE] {state_str}")
            if len(state_tokens) > state_budget:
                # Preserve trailing "| Goals: ..." definition when truncating long JSON state bodies
                if " | Goals: " in state_str:
                    body_part, goals_part = state_str.split(" | Goals: ", 1)
                    goals_tokens = self._encode_segment(f" | Goals: {goals_part}")
                    if len(goals_tokens) < state_budget - 8:
                        body_budget = state_budget - len(goals_tokens)
                        body_tokens = self._encode_segment(f"[STATE] {body_part}")[:body_budget]
                        state_tokens = body_tokens + goals_tokens
                    else:
                        state_tokens = state_tokens[:state_budget]
                else:
                    state_tokens = state_tokens[:state_budget]

            seq_ids: List[int] = list(state_tokens)
            spans: List[Tuple[int, int]] = []
            valid_flags: List[bool] = []

            for blk in cand_token_blocks:
                start_idx = len(seq_ids)
                seq_ids.extend(blk)
                end_idx = len(seq_ids)
                spans.append((start_idx, end_idx))
                valid_flags.append(True)

            # Pad missing candidate slots up to max_candidates pointing safely at token 0
            while len(spans) < max_candidates:
                spans.append((0, 1))
                valid_flags.append(False)

            seq_ids.extend(decide_tokens)
            decide_idx = len(seq_ids) - 1

            all_token_ids.append(seq_ids)
            all_spans.append(spans)
            all_decide_pos.append(decide_idx)
            all_cand_valid.append(valid_flags)

        max_len = max(len(seq) for seq in all_token_ids)
        input_ids = torch.full(
            (batch_size, max_len), fill_value=self._pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for b_idx, seq in enumerate(all_token_ids):
            seq_len = len(seq)
            input_ids[b_idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[b_idx, :seq_len] = 1

        candidate_spans = torch.tensor(all_spans, dtype=torch.long)
        decide_positions = torch.tensor(all_decide_pos, dtype=torch.long)
        candidate_mask = torch.tensor(all_cand_valid, dtype=torch.bool)

        packed = PackedDecisionBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_spans=candidate_spans,
            decide_positions=decide_positions,
            candidate_mask=candidate_mask,
        )
        return packed.to(device) if device is not None else packed


@dataclass(frozen=True)
class BackboneReadoutFeatures:
    """Pooled hidden-state representations extracted by `QwenDecisionBackbone`.

    Attributes:
        decide_hidden: Pooled `[DECIDE]` anchor hidden states of shape `[B, d_backbone]`.
        candidate_hidden: Pooled candidate span hidden states of shape `[B, K, d_backbone]`.
        candidate_mask: Boolean candidate validity mask of shape `[B, K]`.
    """

    decide_hidden: torch.Tensor
    candidate_hidden: torch.Tensor
    candidate_mask: torch.Tensor


class QwenDecisionBackbone(nn.Module):
    """Single-forward-pass Qwen transformer backbone with stripped `lm_head` and LoRA.

    Loads only the base transformer tower (`Qwen3Model` / `Qwen2Model` / `Qwen3_5TextModel`)
    in `bfloat16`, completely eliminating the $V \\times d_{\\text{model}}$ (`151,936 x 1,024`
    to `248,320 x 1,024`) vocabulary projection matrix and logit tensor allocation.
    Attaches `peft` LoRA adapters (`r=16, lora_alpha=32`) onto attention and MLP linear
    projections and extracts pooled latent vectors for `[DECIDE]` and each candidate span.
    """

    def __init__(
        self,
        config: BackboneConfig,
        pretrained_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._config = config

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
            )
        self._tokenizer = tokenizer
        self._packer = DecisionSequencePacker(
            tokenizer=self._tokenizer,
            max_seq_length=config.max_seq_length,
        )

        if pretrained_model is None:
            base_transformer, hidden_size = self._load_and_strip_pretrained(config)
        else:
            base_transformer, hidden_size = self._strip_unneeded_heads(pretrained_model)

        self._hidden_size: int = hidden_size

        if config.gradient_checkpointing and hasattr(
            base_transformer, "gradient_checkpointing_enable"
        ):
            base_transformer.gradient_checkpointing_enable()

        if config.use_lora:
            if not HAS_PEFT:
                raise RuntimeError(
                    "peft is required when BackboneConfig.use_lora=True. Install `peft>=0.15`."
                )
            resolved_targets = self._discover_lora_targets(base_transformer, config.target_modules)
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=list(resolved_targets),
            )
            self._transformer: nn.Module = get_peft_model(base_transformer, lora_cfg)
        else:
            self._transformer = base_transformer

    @classmethod
    def _load_and_strip_pretrained(cls, config: BackboneConfig) -> Tuple[nn.Module, int]:
        """Load the pretrained Qwen base transformer in `config.torch_dtype` without `lm_head`."""
        hf_cfg = AutoConfig.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        dtype = config.resolved_torch_dtype()

        # If checkpoint is a multimodal conditional-generation wrapper (e.g. Qwen3.5-0.8B),
        # load via AutoModel and extract only the pure text transformer tower.
        raw_model = AutoModel.from_pretrained(
            config.model_name_or_path,
            config=hf_cfg,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
        return cls._strip_unneeded_heads(raw_model)

    @staticmethod
    def _strip_unneeded_heads(model: nn.Module) -> Tuple[nn.Module, int]:
        """Strip `lm_head`, multimodal vision towers, and multi-token prediction heads.

        Guarantees zero VRAM allocation for token generation logits or unused vision encoders.
        """
        # Unwrap text tower if nested inside a multimodal or CausalLM container
        if hasattr(model, "language_model") and isinstance(getattr(model, "language_model"), nn.Module):
            base = getattr(model, "language_model")
        elif (
            hasattr(model, "model")
            and isinstance(getattr(model, "model"), nn.Module)
            and hasattr(model, "lm_head")
        ):
            base = getattr(model, "model")
        else:
            base = model

        # Explicitly remove any residual lm_head, visual encoder, or MTP head attributes
        for attr_name in ("lm_head", "visual", "vision_tower", "multi_modal_projector", "mtp"):
            if hasattr(base, attr_name):
                try:
                    delattr(base, attr_name)
                except AttributeError:
                    setattr(base, attr_name, None)
            if base is not model and hasattr(model, attr_name):
                try:
                    delattr(model, attr_name)
                except AttributeError:
                    setattr(model, attr_name, None)

        # Disable KV-cache allocation since single-pass decision heads never decode tokens
        if hasattr(base, "config") and hasattr(base.config, "use_cache"):
            base.config.use_cache = False

        # Resolve hidden dimension across standard Qwen2/Qwen3 and nested text_config schemas
        cfg = getattr(base, "config", None)
        hidden_size: Optional[int] = getattr(cfg, "hidden_size", None)
        if hidden_size is None and cfg is not None and hasattr(cfg, "text_config"):
            hidden_size = getattr(cfg.text_config, "hidden_size", None)
        if hidden_size is None:
            for module in base.modules():
                if isinstance(module, nn.Embedding):
                    hidden_size = int(module.embedding_dim)
                    break
        if hidden_size is None:
            raise ValueError("Unable to infer transformer hidden_size from loaded model.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return base, int(hidden_size)

    @staticmethod
    def _discover_lora_targets(
        model: nn.Module,
        requested_targets: Sequence[str],
    ) -> Tuple[str, ...]:
        """Resolve linear module suffixes present in `model` for `peft` LoRA attachment."""
        linear_suffixes = {
            name.split(".")[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
        }
        matched = tuple(t for t in requested_targets if t in linear_suffixes)
        if matched:
            return matched
        # Fallback for hybrid architectures (e.g. Qwen3.5 linear attention projections)
        fallback = tuple(sorted(s for s in linear_suffixes if s != "lm_head"))
        if not fallback:
            raise ValueError("No nn.Linear layers discovered in backbone for LoRA injection.")
        return fallback

    @property
    def config(self) -> BackboneConfig:
        """Return the immutable `BackboneConfig`."""
        return self._config

    @property
    def hidden_size(self) -> int:
        """Return the transformer hidden dimension $d_{\\text{backbone}}$."""
        return self._hidden_size

    @property
    def tokenizer(self) -> Any:
        """Return the associated tokenizer."""
        return self._tokenizer

    @property
    def packer(self) -> DecisionSequencePacker:
        """Return the `DecisionSequencePacker` instance."""
        return self._packer

    @property
    def transformer(self) -> nn.Module:
        """Return the underlying (LoRA-wrapped) transformer module."""
        return self._transformer

    def has_lm_head(self) -> bool:
        """Verify whether any `lm_head` vocabulary projection remains attached (must be False)."""
        for name, _ in self._transformer.named_modules():
            if name.endswith("lm_head"):
                return True
        return False

    def pack_texts(
        self,
        state_texts: Sequence[str],
        candidate_descriptions: Sequence[Sequence[str]],
        device: Optional[torch.device | str] = None,
    ) -> PackedDecisionBatch:
        """Pack raw `(state_texts, candidate_descriptions)` into a `PackedDecisionBatch`."""
        return self._packer.pack_batch(state_texts, candidate_descriptions, device=device)

    def _pool_spans(
        self,
        hidden_states: torch.Tensor,
        packed_batch: PackedDecisionBatch,
    ) -> BackboneReadoutFeatures:
        """Pool `[DECIDE]` and per-candidate span representations from `hidden_states`.

        Args:
            hidden_states: Last-layer hidden states of shape `[B, S, d_backbone]`.
            packed_batch: Token span metadata on the same device.

        Returns:
            `BackboneReadoutFeatures` with `decide_hidden` `[B, d_backbone]` and
            `candidate_hidden` `[B, K, d_backbone]`.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. Extract [DECIDE] anchor hidden vector: shape [B, d_backbone]
        decide_idx = packed_batch.decide_positions.clamp(0, seq_len - 1).view(batch_size, 1, 1)
        decide_hidden = torch.gather(
            hidden_states,
            dim=1,
            index=decide_idx.expand(batch_size, 1, hidden_dim),
        ).squeeze(1)

        # 2. Extract candidate representations according to PoolingStrategy: shape [B, K, d_backbone]
        strategy = self._config.pooling_strategy
        spans = packed_batch.candidate_spans
        end_indices = (spans[:, :, 1] - 1).clamp(0, seq_len - 1)
        end_pooled = torch.gather(
            hidden_states,
            dim=1,
            index=end_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )

        if strategy == PoolingStrategy.SPAN_END:
            cand_hidden = end_pooled
        else:
            pos = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len)
            starts = spans[:, :, 0:1].clamp(0, seq_len - 1)
            ends = spans[:, :, 1:2].clamp(1, seq_len)
            span_mask = (pos >= starts) & (pos < ends)  # shape: [B, K, S]
            mask_f = span_mask.to(dtype=hidden_states.dtype)
            denom = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean_pooled = torch.einsum("bks,bsd->bkd", mask_f, hidden_states) / denom
            if strategy == PoolingStrategy.SPAN_MEAN:
                cand_hidden = mean_pooled
            else:
                cand_hidden = 0.5 * (end_pooled + mean_pooled)

        # Zero out padded candidate slots cleanly
        cand_mask_f = packed_batch.candidate_mask.unsqueeze(-1).to(dtype=cand_hidden.dtype)
        cand_hidden = cand_hidden * cand_mask_f

        return BackboneReadoutFeatures(
            decide_hidden=decide_hidden,
            candidate_hidden=cand_hidden,
            candidate_mask=packed_batch.candidate_mask,
        )

    def forward(self, packed_batch: PackedDecisionBatch) -> BackboneReadoutFeatures:
        """Run a single transformer forward pass and pool candidate + decision latents."""
        outputs = self._transformer(
            input_ids=packed_batch.input_ids,
            attention_mask=packed_batch.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states: torch.Tensor = outputs.last_hidden_state
        return self._pool_spans(hidden_states, packed_batch)


class BackboneFeatureProjector(nn.Module):
    """FP32 projection neck mapping `bfloat16` backbone states into the AAT latent space.

    Projects `d_backbone` (e.g., `1024` for `Qwen3-0.6B`) into `aat_config.hidden_dim`
    in full `torch.float32` precision with LayerNorm stabilization.
    """

    def __init__(self, backbone_dim: int, head_hidden_dim: int) -> None:
        super().__init__()
        self._norm = nn.LayerNorm(backbone_dim, dtype=torch.float32)
        self._proj = nn.Sequential(
            nn.Linear(backbone_dim, head_hidden_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.LayerNorm(head_hidden_dim, dtype=torch.float32),
        )

    def forward(
        self,
        decide_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cast pooled states to FP32 and project to `[B, d_h]` and `[B, K, d_h]`."""
        dec_fp32 = self._proj(self._norm(decide_hidden.to(torch.float32)))
        cand_fp32 = self._proj(self._norm(candidate_hidden.to(torch.float32)))
        return dec_fp32, cand_fp32


@dataclass(frozen=True)
class TorchAATOutput:
    """Immutable PyTorch autograd container for differentiable AAT transitions at step $t$.

    Attributes:
        rubric_logits: Predicted ordinal rubric logits of shape `[B, K, M, L]` (FP32).
        expected_scores: Expected ordinal goal realizations $\\hat{g}_m(u_k)$ of shape `[B, K, M]`.
        candidate_feasibility: Soft conjunctive satisfaction of $\\vec{a}_{t-1}$ of shape `[B, K]`.
        survival_feasibility: Soft conjunctive satisfaction of $\\vec{a}^{\\min}$ of shape `[B, K]`.
        updated_aspiration: Differentiably adapted aspiration vector $\\hat{\\vec{a}}_t$ of shape `[B, M]`.
        bartleby_prob: Refusal probability $P(\\text{Bartleby} \\mid x_t, \\mathcal{T}_t)$ of shape `[B]`.
        regime_probs: Soft regime distribution `(upward, settled, retreat, bartleby)` `[B, 4]`.
    """

    rubric_logits: torch.Tensor
    expected_scores: torch.Tensor
    candidate_feasibility: torch.Tensor
    survival_feasibility: torch.Tensor
    updated_aspiration: torch.Tensor
    bartleby_prob: torch.Tensor
    regime_probs: torch.Tensor


class TorchPreferenceTableEmbedding(nn.Module):
    """Differentiable FP32 PyTorch encoder for Markov Preference Table $\\mathcal{T}_{t-1}$.

    Embeds $(\\vec{a}_{t-1}, \\vec{a}^{\\min}, \\pi_{t-1}, r_{t-1})$ into $\\mathbb{R}^{d_h}$ while
    preserving gradients with respect to continuous/relaxed `aspiration_prev` inputs during
    Phase 2 multi-step Markov scheduled-sampling rollouts.
    """

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        in_dim = 4 * config.num_goals
        self._net = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim, dtype=torch.float32),
            nn.LayerNorm(config.hidden_dim, dtype=torch.float32),
        )

    def forward(
        self,
        aspiration_prev: torch.Tensor,
        survival_floor: torch.Tensor,
        urgency_ranks: torch.Tensor,
        retreat_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode preference table tensors into dense conditioning vectors `[B, d_h]`."""
        scale = float(max(1, self._config.num_levels - 1))
        norm_a = aspiration_prev.to(torch.float32) / scale
        norm_floor = survival_floor.to(torch.float32) / scale
        norm_ranks = urgency_ranks.to(torch.float32) / float(max(1, self._config.num_goals - 1))

        clamped_retreat = retreat_index.to(torch.long).clamp(0, self._config.num_goals - 1)
        retreat_oh = F.one_hot(clamped_retreat, num_classes=self._config.num_goals).to(torch.float32)

        features = torch.cat([norm_a, norm_floor, norm_ranks, retreat_oh], dim=-1)
        return self._net(features)


class TorchDifferentiableAATModule(nn.Module):
    """FP32 PyTorch differentiable Selten (1998) Aspiration Adaptation and rubric head.

    Evaluates per-candidate ordinal goal distributions across $M$ incomparable goals,
    computes soft conjunctive dominance against $\\vec{a}_{t-1}$ and the Bartleby survival
    floor $\\vec{a}^{\\min}$, and produces a differentiable Markov transition
    $\\mathcal{T}_{t-1} \\to \\hat{\\mathcal{T}}_t$ over the four Selten behavioral regimes:
    `(upward_urgency, settled_feasible, downward_retreat, infeasible_floor)`.

    Context-aware rubric perception (bartleby-1.1):
        The ordinal rubric head consumes only contextual perception features
        ``[candidate_hidden ‖ decide_hidden]``. Preference-table policy state
        (``table_embedding`` / $\\mathcal{T}_{t-1}$) enters later via soft dominance
        gates and the conditioned Bartleby / Selten readout — never as rubric input.
    """

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        out_dim = config.num_goals * config.num_levels
        # Rubric head sees facts from context+candidate only (no table_embedding).
        self._rubric_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, out_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "_level_grid",
            torch.arange(config.num_levels, dtype=torch.float32),
            persistent=False,
        )

    def _compute_expected_scores(self, rubric_logits: torch.Tensor) -> torch.Tensor:
        """Compute expected ordinal realization $\\bar{g}_m(u_k) = \\sum_{\\ell=0}^{L-1} \\ell p_{k,m,\\ell}$."""
        probs = F.softmax(rubric_logits, dim=-1)
        return torch.sum(probs * self._level_grid, dim=-1)

    @staticmethod
    def _compute_soft_dominance(
        expected_scores: torch.Tensor,
        thresholds: torch.Tensor,
        sharpness: float,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute soft conjunctive goal satisfaction $\\prod_{m=1}^M \\sigma(\\gamma (\\bar{g}_{k,m} - \\theta_m))$.

        Uses log-space summation of `logsigmoid` for numerical stability in FP32.
        """
        margin = expected_scores - thresholds.unsqueeze(1)  # shape: [B, K, M]
        log_sat = F.logsigmoid(sharpness * margin).sum(dim=-1)  # shape: [B, K]
        sat = torch.exp(log_sat)
        if candidate_mask is not None:
            sat = sat * candidate_mask.to(dtype=sat.dtype)
        return sat

    def forward(
        self,
        candidate_hidden: torch.Tensor,
        decide_hidden: torch.Tensor,
        aspiration_prev: torch.Tensor,
        survival_floor: torch.Tensor,
        urgency_ranks: torch.Tensor,
        retreat_index: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        override_expected_scores: Optional[torch.Tensor] = None,
        table_embedding: Optional[torch.Tensor] = None,
    ) -> TorchAATOutput:
        """Execute single-pass ordinal rubric grading and differentiable AAT Markov update.

        Args:
            candidate_hidden: Per-candidate pooled features ``[B, K, d_h]``.
            decide_hidden: Context ``[DECIDE]`` pooled features ``[B, d_h]``.
            aspiration_prev: Incoming aspiration vector $\\vec{a}_{t-1}$ ``[B, M]``.
            survival_floor: Survival floor $\\vec{a}^{\\min}$ ``[B, M]``.
            urgency_ranks: Urgency ranks $\\pi_{t-1}$ ``[B, M]``.
            retreat_index: Retreat goal index $r_{t-1}$ ``[B]``.
            candidate_mask: Optional validity mask ``[B, K]``.
            override_expected_scores: Optional teacher ordinal scores ``[B, K, M]``.
            table_embedding: Ignored (retained for call-site compatibility); policy
                conditioning is applied only through soft dominance / readout.
        """
        _ = table_embedding  # Policy state must not enter the rubric perception head.
        batch_size, num_candidates, _ = candidate_hidden.shape
        a_prev = aspiration_prev.to(torch.float32)
        s_floor = survival_floor.to(torch.float32)
        u_ranks = urgency_ranks.to(torch.float32)
        r_idx = retreat_index.to(torch.long).clamp(0, self._config.num_goals - 1)

        # Context-aware rubric: facts from decide+candidate; not from PreferenceTable.
        expanded_decide = decide_hidden.unsqueeze(1).expand(batch_size, num_candidates, -1)
        joint = torch.cat([candidate_hidden, expanded_decide], dim=-1)
        rubric_logits = self._rubric_head(joint).view(
            batch_size, num_candidates, self._config.num_goals, self._config.num_levels
        )

        if override_expected_scores is not None:
            expected_scores = override_expected_scores.to(torch.float32)
        else:
            expected_scores = self._compute_expected_scores(rubric_logits)

        inv_temp = 1.0 / self._config.temperature
        candidate_feas = self._compute_soft_dominance(
            expected_scores, a_prev, sharpness=inv_temp, candidate_mask=candidate_mask
        )
        survival_feas = self._compute_soft_dominance(
            expected_scores,
            s_floor,
            sharpness=self._config.survival_sharpness,
            candidate_mask=candidate_mask,
        )

        # Upward urgency probe along the highest-priority goal (minimum rank index)
        top_urgent = torch.argmin(u_ranks, dim=-1)
        upward_delta = F.one_hot(top_urgent, num_classes=self._config.num_goals).to(torch.float32)
        max_lvl = float(self._config.num_levels - 1)
        upward_aspiration = torch.clamp(a_prev + upward_delta, max=max_lvl)
        upward_feas = self._compute_soft_dominance(
            expected_scores, upward_aspiration, sharpness=inv_temp, candidate_mask=candidate_mask
        )

        # Bounded downward retreat step along r_{t-1}, clamped at survival floor a^{min}
        retreat_delta = F.one_hot(r_idx, num_classes=self._config.num_goals).to(torch.float32)
        retreat_aspiration = torch.maximum(a_prev - retreat_delta, s_floor)

        any_upward = torch.amax(upward_feas, dim=-1)
        any_current = torch.amax(candidate_feas, dim=-1)
        any_survival = torch.amax(survival_feas, dim=-1)

        p_upward = any_upward * any_current
        p_settled = (1.0 - any_upward) * any_current
        p_retreat = (1.0 - any_current) * any_survival
        p_bartleby = (1.0 - any_current) * (1.0 - any_survival)

        regime_stack = torch.stack([p_upward, p_settled, p_retreat, p_bartleby], dim=-1)
        regime_probs = regime_stack / (regime_stack.sum(dim=-1, keepdim=True) + 1e-8)

        updated_aspiration = (
            regime_probs[:, 0:1] * upward_aspiration
            + regime_probs[:, 1:2] * a_prev
            + regime_probs[:, 2:3] * retreat_aspiration
            + regime_probs[:, 3:4] * s_floor
        )

        return TorchAATOutput(
            rubric_logits=rubric_logits,
            expected_scores=expected_scores,
            candidate_feasibility=candidate_feas,
            survival_feasibility=survival_feas,
            updated_aspiration=updated_aspiration,
            bartleby_prob=regime_probs[:, 3],
            regime_probs=regime_probs,
        )


class TorchConditionedNoulHead(nn.Module):
    """FP32 Preference-Table-Conditioned Noul and Bilinear Pointer Readout Head.

    Combines Kev/Clef bilinear pointer compatibility
    $z_k^{\\text{ptr}} = \\frac{1}{\\sqrt{d_p}} k(h_k)^\\top q([h_{\\text{decide}}, e(\\mathcal{T})])$
    with an AAT satisfaction residual conditioned on $\\hat{\\vec{a}}_t$ and $\\vec{a}^{\\min}$
    plus urgency-ordered tie-breaking surplus $\\Delta_{\\text{urg}}(u_k; \\mathcal{T}_t)$.

    Intrinsic Bartleby logit (bartleby-1.1):
        Refusal logit is derived **only** from the soft mathematical floor-violation
        probability approximating $\\forall k:\\,\\exists m\\,\\hat{E}_{k,m} < a_m^{\\min}$,
        scaled by a learned gain. No free residual MLP may shortcut from raw prompt
        keywords into the Bartleby channel.
    """

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        self._query_proj = nn.Linear(config.hidden_dim * 2, config.pointer_dim, dtype=torch.float32)
        self._key_proj = nn.Linear(config.hidden_dim, config.pointer_dim, dtype=torch.float32)
        self._aat_gain = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self._urgency_gain = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # Learned scale only — no residual MLP on decide/table features.
        self._bartleby_gain = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))

    def forward(
        self,
        decide_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        table_embedding: torch.Tensor,
        urgency_ranks: torch.Tensor,
        aat_output: TorchAATOutput,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute conditioned `tradeoff_noul`, `choice_logits`, `choice_probs`, and `bartleby_noul`."""
        q_in = torch.cat([decide_hidden, table_embedding], dim=-1)
        query = self._query_proj(q_in).unsqueeze(-1)  # shape: [B, d_p, 1]
        keys = self._key_proj(candidate_hidden)  # shape: [B, K, d_p]

        scale = 1.0 / math.sqrt(float(self._config.pointer_dim))
        pointer_logits = torch.bmm(keys, query).squeeze(-1) * scale  # shape: [B, K]

        inv_temp = 1.0 / self._config.temperature
        adapted_margin = aat_output.expected_scores - aat_output.updated_aspiration.unsqueeze(1)
        adapted_sat = torch.exp(F.logsigmoid(inv_temp * adapted_margin).sum(dim=-1))

        # Urgency-weighted surplus tie-breaker \Delta_{urg}(u_k; T_t)
        discount = torch.pow(0.65, urgency_ranks.to(torch.float32)).unsqueeze(1)  # [B, 1, M]
        urgency_surplus = (F.relu(adapted_margin) * discount).sum(dim=-1)  # [B, K]

        # Centered floor-margin gating: suppress candidates that violate the survival
        # floor (survival_feasibility -> 0) and center pointer_logits across candidates
        # so slot-0 positional bias cannot override survival-floor compliance.
        floor_gate = aat_output.survival_feasibility  # [B, K] in [0, 1]
        if candidate_mask is not None:
            valid_f = candidate_mask.to(dtype=pointer_logits.dtype)
            ptr_mean = (pointer_logits * valid_f).sum(dim=-1, keepdim=True) / valid_f.sum(
                dim=-1, keepdim=True
            ).clamp(min=1.0)
        else:
            ptr_mean = pointer_logits.mean(dim=-1, keepdim=True)
        centered_pointer = (pointer_logits - ptr_mean) * floor_gate

        effective_sat = adapted_sat * floor_gate
        aat_residual = (
            self._aat_gain * (2.0 * effective_sat - 1.0)
            + self._aat_gain * (2.0 * floor_gate - 1.0)
            + self._urgency_gain * urgency_surplus * floor_gate
        )

        conditioned_logits = centered_pointer + aat_residual
        if candidate_mask is not None:
            conditioned_logits = conditioned_logits.masked_fill(~candidate_mask, -1e9)
            pointer_logits = pointer_logits.masked_fill(~candidate_mask, -1e9)

        # Intrinsic Bartleby: logit of soft ∀k ∃m floor violation; learned scale only.
        clipped_b = aat_output.bartleby_prob.clamp(1e-4, 1.0 - 1e-4)
        bartleby_logit = self._bartleby_gain * torch.log(clipped_b / (1.0 - clipped_b))
        bartleby_noul = torch.sigmoid(bartleby_logit)

        tradeoff_noul = (
            torch.sigmoid(conditioned_logits)
            * floor_gate
            * (1.0 - bartleby_noul.unsqueeze(-1))
        )
        if candidate_mask is not None:
            tradeoff_noul = tradeoff_noul * candidate_mask.to(dtype=tradeoff_noul.dtype)

        choice_logits = torch.cat([conditioned_logits, bartleby_logit.unsqueeze(-1)], dim=-1)
        choice_probs = F.softmax(choice_logits, dim=-1)

        return {
            "pointer_logits": pointer_logits,
            "conditioned_logits": conditioned_logits,
            "choice_logits": choice_logits,
            "choice_probs": choice_probs,
            "tradeoff_noul": tradeoff_noul,
            "bartleby_logit": bartleby_logit,
            "bartleby_noul": bartleby_noul,
        }


class UnconditionedJevReadoutHead(nn.Module):
    """Standard Unconditioned Jev / Kev Readout Head for Controlled Ablations.

    Evaluates `(decide_hidden, candidate_hidden)` **without** any `PreferenceTable`
    ($\\vec{a}_t, \\pi_t, r_t$) or `Bartleby` survival floor ($\\vec{a}^{\\min}$) inputs.
    Used in `StandardJevBaselineSystem` to empirically measure unconditioned noul collapse
    ($P(\\text{yes} \\mid x_t) \\approx 0.5$) on the identical pretrained Qwen backbone.
    """

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        self._query_proj = nn.Linear(config.hidden_dim, config.pointer_dim, dtype=torch.float32)
        self._key_proj = nn.Linear(config.hidden_dim, config.pointer_dim, dtype=torch.float32)
        self._noul_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1, dtype=torch.float32),
        )
        self._abstention_head = nn.Linear(config.hidden_dim, 1, dtype=torch.float32)
        self._rubric_head = nn.Linear(
            config.hidden_dim,
            config.num_goals * config.num_levels,
            dtype=torch.float32,
        )
        self.register_buffer(
            "_level_grid",
            torch.arange(config.num_levels, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        decide_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute unconditioned pointer/noul probabilities from state & candidates alone."""
        batch_size, num_candidates, _ = candidate_hidden.shape
        query = self._query_proj(decide_hidden).unsqueeze(-1)
        keys = self._key_proj(candidate_hidden)
        scale = 1.0 / math.sqrt(float(self._config.pointer_dim))
        pointer_logits = torch.bmm(keys, query).squeeze(-1) * scale

        expanded_dec = decide_hidden.unsqueeze(1).expand(batch_size, num_candidates, -1)
        noul_residual = self._noul_mlp(torch.cat([candidate_hidden, expanded_dec], dim=-1)).squeeze(-1)
        unconditioned_logits = pointer_logits + noul_residual

        if candidate_mask is not None:
            unconditioned_logits = unconditioned_logits.masked_fill(~candidate_mask, -1e9)
            pointer_logits = pointer_logits.masked_fill(~candidate_mask, -1e9)

        bartleby_logit = self._abstention_head(decide_hidden).squeeze(-1)
        bartleby_noul = torch.sigmoid(bartleby_logit)

        tradeoff_noul = torch.sigmoid(unconditioned_logits) * (1.0 - bartleby_noul.unsqueeze(-1))
        if candidate_mask is not None:
            tradeoff_noul = tradeoff_noul * candidate_mask.to(dtype=tradeoff_noul.dtype)

        choice_logits = torch.cat([unconditioned_logits, bartleby_logit.unsqueeze(-1)], dim=-1)
        choice_probs = F.softmax(choice_logits, dim=-1)

        rubric_logits = self._rubric_head(candidate_hidden).view(
            batch_size, num_candidates, self._config.num_goals, self._config.num_levels
        )
        expected_scores = torch.sum(F.softmax(rubric_logits, dim=-1) * self._level_grid, dim=-1)

        return {
            "pointer_logits": pointer_logits,
            "conditioned_logits": unconditioned_logits,
            "choice_logits": choice_logits,
            "choice_probs": choice_probs,
            "tradeoff_noul": tradeoff_noul,
            "bartleby_logit": bartleby_logit,
            "bartleby_noul": bartleby_noul,
            "rubric_logits": rubric_logits,
            "expected_scores": expected_scores,
        }


class BaseDecisionNeuralSystem(nn.Module, ABC):
    """Abstract base class for Qwen-backed single-pass neural decision systems.

    Provides shared optimizer parameter-group partitioning (separating `bfloat16` LoRA
    adapter weights from FP32 readout head weights), trainable parameter accounting,
    and lightweight `safetensors` checkpoint persistence.
    """

    def __init__(
        self,
        backbone: QwenDecisionBackbone,
        aat_config: AATModelConfig,
    ) -> None:
        super().__init__()
        self._backbone = backbone
        self._aat_config = aat_config
        self._projector = BackboneFeatureProjector(
            backbone_dim=backbone.hidden_size,
            head_hidden_dim=aat_config.hidden_dim,
        )

    @property
    @abstractmethod
    def system_type(self) -> str:
        """Unique identifier for checkpoint metadata (`'tradeoff_jev'` or `'standard_jev_baseline'`)."""

    @property
    def backbone(self) -> QwenDecisionBackbone:
        """Return the shared `QwenDecisionBackbone`."""
        return self._backbone

    @property
    def aat_config(self) -> AATModelConfig:
        """Return the `AATModelConfig` governing FP32 readout head dimensions."""
        return self._aat_config

    @abstractmethod
    def _head_module_dict(self) -> Mapping[str, nn.Module]:
        """Return mapping of named FP32 head submodules (including `_projector`)."""

    def build_optimizer_param_groups(
        self,
        lora_lr: float = 1e-4,
        head_lr: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Partition trainable parameters into separate LoRA (`1e-4`) and FP32 Head (`3e-4`) groups.

        Applies weight decay only to matrix weights (`ndim >= 2`) and excludes biases and
        `LayerNorm` / scalar gain parameters (`weight_decay = 0.0`).
        """
        no_decay_tokens = (
            "bias",
            "norm",
            "layernorm",
            "_aat_gain",
            "_urgency_gain",
            "_bartleby_gain",
        )

        lora_decay: List[nn.Parameter] = []
        lora_no_decay: List[nn.Parameter] = []
        for name, param in self._backbone.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(tok in name.lower() for tok in no_decay_tokens):
                lora_no_decay.append(param)
            else:
                lora_decay.append(param)

        head_decay: List[nn.Parameter] = []
        head_no_decay: List[nn.Parameter] = []
        for mod_name, mod in self._head_module_dict().items():
            for p_name, param in mod.named_parameters():
                if not param.requires_grad:
                    continue
                full_name = f"{mod_name}.{p_name}".lower()
                if param.ndim < 2 or any(tok in full_name for tok in no_decay_tokens):
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)

        groups: List[Dict[str, Any]] = []
        if lora_decay:
            groups.append(
                {"params": lora_decay, "lr": lora_lr, "weight_decay": weight_decay, "group_name": "lora_decay"}
            )
        if lora_no_decay:
            groups.append(
                {"params": lora_no_decay, "lr": lora_lr, "weight_decay": 0.0, "group_name": "lora_no_decay"}
            )
        if head_decay:
            groups.append(
                {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay, "group_name": "head_decay"}
            )
        if head_no_decay:
            groups.append(
                {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "group_name": "head_no_decay"}
            )
        return groups

    def trainable_parameter_summary(self) -> Dict[str, int]:
        """Compute total, frozen backbone, trainable LoRA, and trainable FP32 head parameter counts."""
        backbone_total = sum(p.numel() for p in self._backbone.parameters())
        lora_trainable = sum(p.numel() for p in self._backbone.parameters() if p.requires_grad)
        head_trainable = sum(
            p.numel()
            for mod in self._head_module_dict().values()
            for p in mod.parameters()
            if p.requires_grad
        )
        return {
            "total_parameters": backbone_total + head_trainable,
            "backbone_frozen_parameters": backbone_total - lora_trainable,
            "lora_trainable_parameters": lora_trainable,
            "head_fp32_trainable_parameters": head_trainable,
            "all_trainable_parameters": lora_trainable + head_trainable,
        }

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Persist LoRA adapter weights + FP32 head `safetensors` + JSON configuration.

        Avoids duplicating the frozen 1.2–1.5 GB base Qwen checkpoint on disk so each
        saved checkpoint occupies only a few megabytes.
        """
        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save LoRA adapter weights (or lightweight backbone trainable state)
        if HAS_PEFT and isinstance(self._backbone.transformer, PeftModel):
            adapter_dir = out_dir / "lora_adapter"
            self._backbone.transformer.save_pretrained(str(adapter_dir), safe_serialization=True)
        else:
            trainable_backbone = {
                k: v.detach().cpu().contiguous()
                for k, v in self._backbone.state_dict().items()
            }
            save_safetensors(trainable_backbone, str(out_dir / "backbone_state.safetensors"))

        # 2. Save FP32 readout head weights via safetensors
        head_tensors: Dict[str, torch.Tensor] = {}
        for mod_name, mod in self._head_module_dict().items():
            for k, v in mod.state_dict().items():
                head_tensors[f"{mod_name}.{k}"] = v.detach().cpu().to(torch.float32).contiguous()
        save_safetensors(head_tensors, str(out_dir / "aat_head.safetensors"))

        # 3. Save metadata & configs
        metadata: Dict[str, Any] = {
            "system_type": self.system_type,
            "backbone_config": self._backbone.config.to_dict(),
            "aat_config": asdict(self._aat_config),
            "parameter_summary": self.trainable_parameter_summary(),
        }
        if extra_metadata is not None:
            metadata["extra"] = dict(extra_metadata)
        (out_dir / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return out_dir

    def load_checkpoint_weights(
        self,
        checkpoint_dir: str | Path,
        map_location: Optional[torch.device | str] = None,
    ) -> Mapping[str, Any]:
        """Load saved LoRA adapter weights and FP32 head `safetensors` into this instance."""
        ckpt_dir = Path(checkpoint_dir)
        meta_path = ckpt_dir / "checkpoint_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing checkpoint_metadata.json in {ckpt_dir}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        ckpt_sys_type = metadata.get("system_type")
        if self.system_type in {"tradeoff_jev", "bartleby-1.0", "bartleby-1.1", "bartleby-1.2"}:
            compatible_types = {"tradeoff_jev", "bartleby-1.0", "bartleby-1.1", "bartleby-1.2"}
        elif self.system_type in {"standard_jev_baseline", "prompt_conditioned_jev_baseline"}:
            compatible_types = {"standard_jev_baseline", "prompt_conditioned_jev_baseline"}
        else:
            compatible_types = {self.system_type}
        if ckpt_sys_type not in compatible_types:
            raise ValueError(
                f"Checkpoint system_type {ckpt_sys_type!r} does not match "
                f"instance system_type {self.system_type!r}."
            )

        target_dev = str(map_location) if map_location is not None else "cpu"

        adapter_dir = ckpt_dir / "lora_adapter"
        adapter_sf = adapter_dir / "adapter_model.safetensors"
        if HAS_PEFT and isinstance(self._backbone.transformer, PeftModel) and adapter_sf.exists():
            from peft import set_peft_model_state_dict

            adapter_state = load_safetensors(str(adapter_sf), device=target_dev)
            set_peft_model_state_dict(self._backbone.transformer, adapter_state)
        elif (ckpt_dir / "backbone_state.safetensors").exists():
            bb_state = load_safetensors(str(ckpt_dir / "backbone_state.safetensors"), device=target_dev)
            self._backbone.load_state_dict(bb_state, strict=False)

        head_sf = (
            ckpt_dir / "aat_head.safetensors"
            if (ckpt_dir / "aat_head.safetensors").exists()
            else ckpt_dir / "heads.safetensors"
        )
        head_state = load_safetensors(str(head_sf), device=target_dev)
        for mod_name, mod in self._head_module_dict().items():
            prefix = f"{mod_name}."
            sub_state = {
                k[len(prefix) :]: v for k, v in head_state.items() if k.startswith(prefix)
            }
            expected_state = mod.state_dict()
            filtered_state: Dict[str, torch.Tensor] = {}
            for k, v in sub_state.items():
                if k in expected_state and expected_state[k].shape == v.shape:
                    filtered_state[k] = v
            if len(filtered_state) == len(expected_state):
                mod.load_state_dict(filtered_state, strict=True)
            else:
                mod.load_state_dict(filtered_state, strict=False)

        return metadata


class TradeOffJevNeuralSystem(BaseDecisionNeuralSystem):
    """End-to-end single-pass Trade-Off Jev Neural System (`QwenDecisionBackbone` + FP32 AAT Head).

    Integrates `QwenDecisionBackbone` (`bfloat16` + LoRA) with `TorchPreferenceTableEmbedding`,
    `TorchDifferentiableAATModule`, and `TorchConditionedNoulHead` in FP32. Supports both
    Phase 1 single-step teacher-forced supervision and Phase 2 multi-step Markov scheduled
    sampling where $\\hat{\\vec{a}}_{t-1}$ flows differentiably into step $t$.
    """

    def __init__(
        self,
        backbone: QwenDecisionBackbone,
        aat_config: AATModelConfig,
    ) -> None:
        super().__init__(backbone=backbone, aat_config=aat_config)
        self._table_embedding = TorchPreferenceTableEmbedding(aat_config)
        self._aat_module = TorchDifferentiableAATModule(aat_config)
        self._noul_head = TorchConditionedNoulHead(aat_config)

    @property
    def system_type(self) -> str:
        return "tradeoff_jev"

    def _head_module_dict(self) -> Mapping[str, nn.Module]:
        return {
            "projector": self._projector,
            "table_embedding": self._table_embedding,
            "aat_module": self._aat_module,
            "noul_head": self._noul_head,
        }

    def forward(
        self,
        packed_batch: PackedDecisionBatch,
        aspiration_prev: torch.Tensor,
        survival_floor: torch.Tensor,
        urgency_ranks: torch.Tensor,
        retreat_index: torch.Tensor,
        override_expected_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run single-pass backbone encoding, differentiable AAT transition, and noul readout."""
        features = self._backbone(packed_batch)
        decide_h, cand_h = self._projector(features.decide_hidden, features.candidate_hidden)

        table_emb = self._table_embedding(
            aspiration_prev=aspiration_prev,
            survival_floor=survival_floor,
            urgency_ranks=urgency_ranks,
            retreat_index=retreat_index,
        )
        aat_out = self._aat_module(
            candidate_hidden=cand_h,
            decide_hidden=decide_h,
            aspiration_prev=aspiration_prev,
            survival_floor=survival_floor,
            urgency_ranks=urgency_ranks,
            retreat_index=retreat_index,
            candidate_mask=features.candidate_mask,
            override_expected_scores=override_expected_scores,
        )
        readout = self._noul_head(
            decide_hidden=decide_h,
            candidate_hidden=cand_h,
            table_embedding=table_emb,
            urgency_ranks=urgency_ranks,
            aat_output=aat_out,
            candidate_mask=features.candidate_mask,
        )
        return {
            **readout,
            "rubric_logits": aat_out.rubric_logits,
            "expected_scores": aat_out.expected_scores,
            "candidate_feasibility": aat_out.candidate_feasibility,
            "survival_feasibility": aat_out.survival_feasibility,
            "updated_aspiration": aat_out.updated_aspiration,
            "regime_probs": aat_out.regime_probs,
        }


def format_policy_conditioned_state_text(
    state_text: str,
    aspiration_prev: Sequence[float],
    survival_floor: Sequence[float],
    urgency_ranks: Sequence[float],
    retreat_index: int,
) -> str:
    """Append serialized `SeltenPreferenceTable` ($\\vec{a}_{t-1}, \\vec{a}^{\\min}, \\pi_{t-1}, r_{t-1}$) to `state_text`.

    Used by `PromptConditionedJevBaselineSystem` to test unstructured text-prompt
    policy conditioning on the identical `Qwen/Qwen3-0.6B` + `(K+1)`-way
    `UnconditionedJevReadoutHead` architecture without `PreferenceTableEncoder`,
    `SeltenAdaptationHead`, or multiplicative survival-floor gating.
    """
    if " | Policy: " in state_text:
        return state_text
    parts: List[str] = []
    n_goals = min(len(aspiration_prev), len(survival_floor), len(urgency_ranks))
    for idx in range(n_goals):
        ret_tag = " [RET]" if idx == int(retreat_index) else ""
        parts.append(
            f"g{idx}(a_prev={float(aspiration_prev[idx]):.1f}, "
            f"floor={float(survival_floor[idx]):.1f}, "
            f"rank={int(round(float(urgency_ranks[idx])))}){ret_tag}"
        )
    return f"{state_text} | Policy: {'; '.join(parts)}"


class StandardJevBaselineSystem(BaseDecisionNeuralSystem):
    """Controlled Baseline System (`QwenDecisionBackbone` + `UnconditionedJevReadoutHead`).

    Uses the exact same `QwenDecisionBackbone` and LoRA configuration as
    `TradeOffJevNeuralSystem`, but omits `PreferenceTableEncoder`, `SeltenAdaptationHead`,
    and multiplicative survival-floor gating. When `prompt_conditioned=True`
    (or via `PromptConditionedJevBaselineSystem`), the active `SeltenPreferenceTable`
    is serialized into the text prompt (`| Policy: g0(...)`) fed to the Qwen encoder.
    """

    def __init__(
        self,
        backbone: QwenDecisionBackbone,
        aat_config: AATModelConfig,
        *,
        prompt_conditioned: bool = False,
    ) -> None:
        super().__init__(backbone=backbone, aat_config=aat_config)
        self._unconditioned_head = UnconditionedJevReadoutHead(aat_config)
        self._prompt_conditioned = bool(prompt_conditioned)

    @property
    def prompt_conditioned(self) -> bool:
        """Whether `SeltenPreferenceTable` is serialized into the text prompt."""
        return self._prompt_conditioned

    @property
    def system_type(self) -> str:
        return (
            "prompt_conditioned_jev_baseline"
            if self._prompt_conditioned
            else "standard_jev_baseline"
        )

    def _head_module_dict(self) -> Mapping[str, nn.Module]:
        return {
            "projector": self._projector,
            "unconditioned_head": self._unconditioned_head,
        }

    def forward(
        self,
        packed_batch: PackedDecisionBatch,
        aspiration_prev: Optional[torch.Tensor] = None,
        survival_floor: Optional[torch.Tensor] = None,
        urgency_ranks: Optional[torch.Tensor] = None,
        retreat_index: Optional[torch.Tensor] = None,
        override_expected_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run readout head without algebraic AAT or survival-floor gating."""
        _ = (aspiration_prev, survival_floor, urgency_ranks, retreat_index, override_expected_scores)
        features = self._backbone(packed_batch)
        decide_h, cand_h = self._projector(features.decide_hidden, features.candidate_hidden)
        return self._unconditioned_head(
            decide_hidden=decide_h,
            candidate_hidden=cand_h,
            candidate_mask=features.candidate_mask,
        )


class PromptConditionedJevBaselineSystem(StandardJevBaselineSystem):
    """Prompt-Conditioned `(K+1)`-Way Standard Jev Baseline System.

    Serializes the incoming `SeltenPreferenceTable` ($\\vec{a}_{t-1}, \\vec{a}^{\\min}, \\pi_{t-1}, r_{t-1}$)
    directly into the text prompt (`| Policy: g0(a_prev=..., floor=..., rank=...) [RET]`)
    processed by `QwenDecisionBackbone` + `UnconditionedJevReadoutHead`, testing how far
    unstructured prompt conditioning reaches without an epistemic-normative firewall,
    `SeltenAdaptationHead`, or multiplicative survival-floor gating.
    """

    def __init__(
        self,
        backbone: QwenDecisionBackbone,
        aat_config: AATModelConfig,
    ) -> None:
        super().__init__(backbone=backbone, aat_config=aat_config, prompt_conditioned=True)


class SeltenAdaptationHead(TorchDifferentiableAATModule):
    """First-class Reinhard Selten (1998, 1999) Differentiable Aspiration Adaptation Head.

    Evaluates per-candidate ordinal rubric distributions across $M$ incomparable goals,
    computes soft conjunctive dominance against $\\vec{a}_{t-1}$ and the Bartleby survival
    floor $\\vec{a}^{\\min}$, and produces a differentiable Markov transition
    $\\mathcal{T}_{t-1} \\to \\hat{\\mathcal{T}}_t$.
    """


class TorchPreferenceConditionedNoulHead(TorchConditionedNoulHead):
    """First-class Preference-Table-Conditioned Noul and Bilinear Pointer Readout Head."""


class BartlebyReadoutHead(TorchConditionedNoulHead):
    """First-class Bartleby Survival-Floor & Preference-Conditioned Noul Readout Head.

    Computes calibrated `tradeoff_noul` $P(\\text{yes} \\mid x_t, u_k, \\mathcal{T}_t)$,
    $(K+1)$-way `choice_logits`, and intrinsic `bartleby_noul`
    $P(\\text{\"I would prefer not to\"} \\mid x_t, \\mathcal{T}_t)$ in a single forward pass.
    """


class BartlebyNeuralSystem(TradeOffJevNeuralSystem):
    """First-class `bartleby-1.1` Single-Pass Neural System (`QwenDecisionBackbone` + Selten & Bartleby Heads)."""

