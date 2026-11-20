import random
import sysconfig
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from seventy.api.prefix_tokens import PrefixTokenizer, common_prefix


def source_text(n: int, seed: int) -> str:
    files = sorted(p for p in Path(sysconfig.get_paths()["stdlib"]).glob("*.py") if p.stat().st_size > 4000)
    r = random.Random(seed)
    out = ""
    while len(out) < n:
        src = r.choice(files).read_text(errors="replace")
        start = r.randrange(0, len(src) // 2)
        out += src[start : start + r.randrange(200, 3000)]
    return out[:n]


@pytest.fixture(scope="module")
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=2000, special_tokens=["<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([source_text(200_000, 0)], trainer)
    return tok


def render(messages: list[tuple[str, str]], generation: bool = True) -> str:
    text = "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in messages)
    return text + ("<|im_start|>assistant\n" if generation else "")


def full(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def test_common_prefix():
    a = "x" * 200_000 + "abc"
    assert common_prefix(a, a) == len(a)
    assert common_prefix(a, a[:-1] + "d") == len(a) - 1
    assert common_prefix(a, "x" * 70_000 + "y") == 70_000
    assert common_prefix("", a) == 0


def test_growing_conversation_reuses_tokens_and_matches(tokenizer):
    pt = PrefixTokenizer(tokenizer, min_chars=500)
    r = random.Random(1)
    messages = [("system", "You are a coding agent."), ("user", source_text(3000, 2))]
    for turn in range(12):
        text = render(messages)
        assert pt.encode(text) == full(tokenizer, text)
        messages.append(("assistant", source_text(r.randrange(50, 2000), 10 + turn)))
        messages.append(("tool" if turn % 2 else "user", source_text(r.randrange(50, 4000), 100 + turn)))
    assert pt.reused >= 10


def test_edits_branches_and_markers_inside_content(tokenizer):
    pt = PrefixTokenizer(tokenizer, min_chars=500, entries=4)
    base = [("system", "s"), ("user", source_text(5000, 3)), ("assistant", source_text(800, 4))]
    cases = [
        render(base),
        render(base + [("user", "next question")]),
        render(base[:2] + [("assistant", source_text(800, 5))]),
        render(base + [("user", "a literal <|im_start|> marker inside the text")]),
        render(base + [("user", "a literal <|im_start|> marker inside the text, longer")]),
        render(base, generation=False) + "<|im_start|>user\npartial",
        render(base, generation=False) + "<|im_start|",
        render([("system", "t")] + base[1:]),
        render(base)[:-5] + "zzzzz" + render(base + [("user", "q")])[len(render(base)):],
    ]
    for _ in range(2):
        for text in cases:
            assert pt.encode(text) == full(tokenizer, text)


def test_short_texts_are_not_cached(tokenizer):
    pt = PrefixTokenizer(tokenizer, min_chars=10_000)
    text = render([("user", "short")])
    assert pt.encode(text) == full(tokenizer, text)
    assert not pt.entries


@pytest.mark.model
def test_model_tokenizer_on_agent_turns(model_path):
    tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    pt = PrefixTokenizer(tok, min_chars=1000)
    messages = [("system", "You are an agent."), ("user", source_text(20_000, 7))]
    for turn in range(6):
        text = render(messages)
        assert pt.encode(text) == full(tok, text)
        messages += [("assistant", source_text(1500, 20 + turn) + " é 漢字 🙂"), ("tool", source_text(5000, 30 + turn))]
    assert pt.reused >= 5
