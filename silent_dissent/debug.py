"""A tiny randomly initialised Llama + word-level tokenizer for offline pipeline checks.

    python -m silent_dissent.debug results/debug_model   # then use configs/debug.yaml
"""
from __future__ import annotations

import sys

from .prompts import (ANSWER_PREFIX, GENERIC_REASON, SYSTEM_PROMPT, WITHHELD_TURN, format_peer_message,
                      format_question)

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def make_tokenizer(items):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    corpus = [format_question(it) for it in items] + [
        format_peer_message(["x"]), ANSWER_PREFIX, SYSTEM_PROMPT, GENERIC_REASON, WITHHELD_TURN,
        "A B C D E . <|system|> <|user|> <|assistant|> I briefly considered but in the end : Agent 2 3 4 5 6",
        "Argue in one or two sentences that the correct answer is Do not mention any other option and do not "
        "restate the letter.",
    ]
    pre = pre_tokenizers.Whitespace()
    words = {w for text in corpus for w, _ in pre.pre_tokenize_str(text)}
    vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]", *sorted(words)])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[PAD]")
    tok.chat_template = TEMPLATE
    tok.padding_side = "left"
    return tok


def make_model(tok, n_layers: int = 4, seed: int = 0):
    import torch
    import transformers

    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=4096, pad_token_id=tok.pad_token_id)
    return transformers.LlamaForCausalLM(cfg).eval()


if __name__ == "__main__":
    from pathlib import Path

    from .data import load_items

    out = sys.argv[1] if len(sys.argv) > 1 else "results/debug_model"
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "items.jsonl"
    tok = make_tokenizer(load_items(str(fixture)))
    model = make_model(tok)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved debug model to {out}")
