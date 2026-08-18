r"""
Tier-3 Banglish transliteration: the model, and the runtime `tier3_fn`
`bntok.banglish.transliterate()` is built to accept.

The model class lives here, not in `scripts/train_banglish_translit.py`
where it was first written, because `bntok` (the library) needs it too -
`load_tier3_fn()` below is what actually wires a trained checkpoint into
`transliterate()`'s pluggable hook. `scripts/train_banglish_translit.py`
and `scripts/eval_banglish_translit.py` import `TranslitTransformer` and
the vocabulary constants from here now, so there is exactly one
definition, not two that could drift apart - a library should not import
from `scripts/`, and moving the model here is what fixes that direction
rather than reaching into `scripts/` from a pip-installed package.

torch is an optional dependency of this module specifically (not of
`bntok` as a whole - the core tokenizer needs none of it), imported lazily
inside `load_tier3_fn()` so `import bntok.banglish` and plain tier-0/1/2
Banglish resolution never pay for it.
"""

from __future__ import annotations

import math

from . import substrate
from .akshara import aksharas
from .errors import ConfigError

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

# Diacritics that must be attached to a preceding Consonant/Vowel chunk to be
# well-formed - a decoded string where one of these starts its own "other"
# fallback chunk (aksharas() never raises, but records it as a plain
# grapheme cluster rather than absorbing it into an akshara) is an orphan
# mark: the model emitted a virama/matra/modifier with no valid base, a
# structural defect a from-scratch char-level decoder can produce that pure
# beam score does not penalize. Used by _conjunct_violations below to
# re-rank otherwise-close beam candidates toward the well-formed one.
_ORPHAN_STARTERS = substrate.MATRAS | substrate.MODIFIERS | {substrate.VIRAMA, substrate.NUKTA}


def _conjunct_violations(text: str) -> int:
    """Count orphan-diacritic chunks in a decoded string via the same
    grammar-first parser the rest of this project trusts for Bengali
    structure (bntok.akshara), not a new ad-hoc validator. Zero for any
    well-formed Bengali (or non-Bengali) text; only counts real defects.
    """
    return sum(
        1 for a in aksharas(text)
        if a.kind == "other" and a.text and a.text[0] in _ORPHAN_STARTERS
    )


def _build_model_classes():
    """Defines PositionalEncoding/TranslitTransformer against a real torch
    import - deferred into a function so importing this module at all does
    not require torch to be installed. Called once, lazily, by
    load_tier3_fn(); scripts/ that already depend on torch unconditionally
    call this directly to get the classes for training/eval.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1)]

    class TranslitTransformer(nn.Module):
        """A small from-scratch Transformer encoder-decoder. Character-level
        both sides: no subword vocabulary, so no OOV problem on noisy Latin
        spelling variance or digit-substitution slang.
        """

        def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256,
                     nhead: int = 8, num_layers: int = 4, dim_ff: int = 1024, dropout: float = 0.1,
                     src_pad: int = 0, tgt_pad: int = 0):
            super().__init__()
            self.d_model = d_model
            self.src_pad = src_pad
            self.tgt_pad = tgt_pad
            self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad)
            self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad)
            self.pos_enc = PositionalEncoding(d_model)
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                num_decoder_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout,
                batch_first=True,
            )
            self.out_proj = nn.Linear(d_model, tgt_vocab_size)

        def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
            src_key_padding = (src == self.src_pad)
            tgt_key_padding = (tgt_in == self.tgt_pad)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(src.device)
            s = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
            t = self.pos_enc(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))
            out = self.transformer(
                s, t, tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding, tgt_key_padding_mask=tgt_key_padding,
                memory_key_padding_mask=src_key_padding,
            )
            return self.out_proj(out)

        @torch.no_grad()
        def greedy_decode(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32) -> torch.Tensor:
            self.eval()
            device = src.device
            batch = src.size(0)
            src_key_padding = (src == self.src_pad)
            s = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(s, src_key_padding_mask=src_key_padding)
            ys = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
            done = torch.zeros(batch, dtype=torch.bool, device=device)
            for _ in range(max_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
                t = self.pos_enc(self.tgt_embed(ys) * math.sqrt(self.d_model))
                out = self.transformer.decoder(t, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding)
                logits = self.out_proj(out[:, -1, :])
                next_tok = logits.argmax(-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                done = done | (next_tok.squeeze(1) == eos_id)
                if done.all():
                    break
            return ys

        @torch.no_grad()
        def beam_decode(self, src_row: torch.Tensor, bos_id: int, eos_id: int,
                         beam_size: int = 5, max_len: int = 32, length_penalty: float = 0.6,
                         tgt_vocab: list | None = None) -> list:
            """Beam search for ONE example (src_row: shape (1, src_len)). Greedy
            commits to the single best character at each step and can't recover
            from an early wrong pick; beam search keeps `beam_size` candidate
            sequences alive and picks the best-scoring complete one at the end.

            `tgt_vocab` (id -> char list), if given, turns on a structural
            re-rank pass over the surviving beams: among the `beam_size` kept
            candidates, prefer the one with the fewest orphan-diacritic
            violations (see _conjunct_violations), breaking ties by the
            original length-normalized score. Beam score alone has no notion
            of "valid Bengali conjunct" - this costs nothing extra (no new
            model calls, just re-ordering candidates already computed) and
            only changes the pick when it recovers a well-formed runner-up
            the raw score alone would have passed over. None (default)
            reproduces the old score-only behavior exactly.
            """
            self.eval()
            device = src_row.device
            src_key_padding = (src_row == self.src_pad)
            s = self.pos_enc(self.src_embed(src_row) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(s, src_key_padding_mask=src_key_padding)

            def norm_score(item):
                score, seq, _ = item
                return score / (len(seq) ** length_penalty)

            beams = [(0.0, [bos_id], False)]
            for _ in range(max_len):
                candidates = []
                for score, seq, finished in beams:
                    if finished:
                        candidates.append((score, seq, finished))
                        continue
                    ys = torch.tensor([seq], dtype=torch.long, device=device)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
                    t = self.pos_enc(self.tgt_embed(ys) * math.sqrt(self.d_model))
                    out = self.transformer.decoder(t, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding)
                    logprobs = F.log_softmax(self.out_proj(out[:, -1, :]), dim=-1).squeeze(0)
                    topk_logprobs, topk_ids = logprobs.topk(min(beam_size, logprobs.size(-1)))
                    for lp, tid in zip(topk_logprobs.tolist(), topk_ids.tolist()):
                        candidates.append((score + lp, seq + [tid], tid == eos_id))
                candidates.sort(key=norm_score, reverse=True)
                beams = candidates[:beam_size]
                if all(b[2] for b in beams):
                    break

            beams.sort(key=norm_score, reverse=True)
            if tgt_vocab is None:
                return beams[0][1]

            def violations(item):
                seq = item[1]
                text = "".join(tgt_vocab[i] for i in seq if i not in (bos_id, eos_id))
                return _conjunct_violations(text)

            beams.sort(key=lambda item: (violations(item), -norm_score(item)))
            return beams[0][1]

        def beam_decode_batch(self, src: torch.Tensor, bos_id: int, eos_id: int,
                               beam_size: int = 5, max_len: int = 32, length_penalty: float = 0.6,
                               tgt_vocab: list | None = None) -> torch.Tensor:
            """Beam search over a batch, one example at a time (see beam_decode),
            padded back into a single tensor with the same (batch, seq_len) shape
            and pad/bos/eos semantics greedy_decode returns.
            """
            seqs = [
                self.beam_decode(src[i:i + 1], bos_id, eos_id, beam_size=beam_size,
                                  max_len=max_len, length_penalty=length_penalty, tgt_vocab=tgt_vocab)
                for i in range(src.size(0))
            ]
            max_seq_len = max(len(s) for s in seqs)
            out = torch.full((len(seqs), max_seq_len), self.tgt_pad, dtype=torch.long, device=src.device)
            for i, s in enumerate(seqs):
                out[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=src.device)
            return out

    return PositionalEncoding, TranslitTransformer


def load_tier3_fn(ckpt_path: str, vocab_path: str, beam_size: int = 5, device: str = "cpu"):
    """Build a real `tier3_fn` (str -> str | None) for
    `bntok.banglish.transliterate()`, from a trained checkpoint.

    beam_size=5 by default, not greedy: measured on the reserved Dakshina
    test split, beam search moved exact-match 45.1% -> 53.9% and CER
    16.9% -> 13.5% on the same checkpoint (docs/known-issues.md) - the
    better decoder should be the default for anyone just wiring this in,
    not an opt-in a caller has to know to ask for.

    The beam's structural re-rank (see TranslitTransformer.beam_decode's
    `tgt_vocab` argument) is on by default here too, for the same reason:
    it costs no extra model calls, only reorders candidates already
    computed, and can only move the pick toward a well-formed conjunct a
    runner-up beam already reached.

    Returns None from the callable (transliterate()'s "I can't resolve
    this either" contract) only on decode failure, never fabricates an
    empty-string result as if it were a real transliteration.
    """
    try:
        import json

        import torch
    except ImportError as e:
        raise ConfigError("tier-3 needs torch: pip install torch") from e

    _, TranslitTransformer = _build_model_classes()

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    src_stoi, tgt_stoi, tgt_vocab = vocab["src_stoi"], vocab["tgt_stoi"], vocab["tgt_vocab"]
    bos_id, eos_id, pad_id = tgt_stoi[BOS], tgt_stoi[EOS], tgt_stoi[PAD]

    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=dev)
    if "config" not in ckpt:
        raise ConfigError(
            f"{ckpt_path} has no saved model config (an older checkpoint format) - "
            "cannot reconstruct the exact architecture it was trained with"
        )
    model = TranslitTransformer(**ckpt["config"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def tier3_fn(word: str) -> str | None:
        if not word:
            return None
        src_ids = [src_stoi.get(c, src_stoi[UNK]) for c in word]
        src = torch.tensor([src_ids], dtype=torch.long, device=dev)
        try:
            pred = model.beam_decode_batch(src, bos_id, eos_id, beam_size=beam_size, max_len=len(word) + 8,
                                            tgt_vocab=tgt_vocab)
        except Exception:  # noqa: BLE001 - a decode failure means "can't resolve", not a crash
            return None
        ids = [c for c in pred[0].tolist() if c not in (pad_id, bos_id, eos_id)]
        if not ids:
            return None
        return "".join(tgt_vocab[i] for i in ids)

    return tier3_fn
