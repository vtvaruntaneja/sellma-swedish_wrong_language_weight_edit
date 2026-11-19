"""Build data/ from the hand-written lists in data/source/ and from wikimedia/wikipedia."""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

DATA = Path(__file__).parent / "data"
SEED = 0
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
N_PARAGRAPHS = 50


def read_tsv(name):
    return [line.rstrip("\n").split("\t") for line in open(DATA / "source" / name, encoding="utf-8") if line.strip()]


def write_jsonl(name, rows):
    with open(DATA / name, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_swedish():
    rows = read_tsv("sv_prompts.tsv")
    by_cat = {}
    for i, (cat, text) in enumerate(rows):
        by_cat.setdefault(cat, []).append({"id": f"sv{i:03d}", "category": cat, "prompt": text})
    rng = random.Random(SEED)
    dev, test = [], []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        rng.shuffle(items)
        dev += items[: len(items) // 2]
        test += items[len(items) // 2:]
    rng.shuffle(dev)
    rng.shuffle(test)
    write_jsonl("sv_dev.jsonl", dev)
    write_jsonl("sv_test.jsonl", test)


def interleave(rows, lang_of):
    """Alternate the two languages so that every prefix of the file is balanced."""
    sv = [r for r in rows if lang_of(r) == "sv"]
    en = [r for r in rows if lang_of(r) == "en"]
    return [r for pair in zip(sv, en) for r in pair]


def english_and_qa():
    write_jsonl("en_test.jsonl", [{"id": f"en{i:03d}", "category": c, "prompt": t}
                                  for i, (c, t) in enumerate(read_tsv("en_prompts.tsv"))])
    write_jsonl("en_dev.jsonl", [{"id": f"ed{i:03d}", "category": c, "prompt": t}
                                 for i, (c, t) in enumerate(read_tsv("en_dev_prompts.tsv"))])
    qa = interleave(read_tsv("qa.tsv"), lambda r: r[0])
    write_jsonl("qa_test.jsonl", [{"id": f"qa{i:02d}", "lang": lang, "question": q, "answers": a.split("|")}
                                  for i, (lang, q, a) in enumerate(qa)])


def good_paragraph(p):
    return (300 <= len(p) <= 1000 and p.count(". ") >= 2 and p.rstrip().endswith(".")
            and not re.search(r"[|=*{}<>\[\]]", p))


def paragraphs(lang, n_articles_needed):
    stream = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train",
                          streaming=True, revision=WIKI_REVISION)
    out = []
    for article in stream:
        para = next((p.strip() for p in article["text"].split("\n") if good_paragraph(p.strip())), None)
        if para:
            out.append({"lang": lang, "title": article["title"], "url": article["url"], "text": para})
        if len(out) == n_articles_needed:
            return out


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


def wikipedia():
    ppl, langid = [], []
    for lang in ["sv", "en"]:
        paras = paragraphs(lang, 2 * N_PARAGRAPHS)
        ppl += paras[:N_PARAGRAPHS]
        langid += paras[N_PARAGRAPHS:]
    for lang in ["no", "da"]:
        langid += paragraphs(lang, N_PARAGRAPHS)
    write_jsonl("wiki_ppl.jsonl", [{"id": f"wp{i:03d}", **p} for i, p in enumerate(interleave(ppl, lambda p: p["lang"]))])
    write_jsonl("langid_check.jsonl", [{"id": f"li{i:03d}", **p, "first_sentence": first_sentence(p["text"])}
                                       for i, p in enumerate(langid)])


if __name__ == "__main__":
    split_swedish()
    english_and_qa()
    wikipedia()
