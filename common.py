"""Shared functions: model loading, generation, language identification, activations, weight edits, scoring."""
import json
import math
from pathlib import Path

import torch
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

ROOT = Path(__file__).parent
MODELS = {
    "qwen35": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "qwen25": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
}
DEVICE = "mps"
BATCH = 32
MAX_NEW_TOKENS = 512
QA_MAX_NEW_TOKENS = 64
# Language rule for the 512-token runs, fixed before any of them was run: a reply is in language X only if both
# halves (split at the middle word) are classified X. Replies under MIN_WORDS_TO_SPLIT words are not split, so each
# half is at least 20 words, about the length of the first sentences lingua was checked on (results/langid_check*).
MIN_WORDS_TO_SPLIT = 40
SYSTEM_SV = "Svara alltid på svenska."
# Recommended sampling, one per model:
# qwen35: model card "Best Practices", non-thinking mode for text tasks.
# qwen25: the card lists none; these are the values in the repo's generation_config.json.
SAMPLING = {
    "qwen35": dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=2.0),
    "qwen25": dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1, presence_penalty=0.0),
}
LANG_CODES = {Language.SWEDISH: "sv", Language.BOKMAL: "no", Language.NYNORSK: "no",
              Language.DANISH: "da", Language.ENGLISH: "en"}


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load(model_key):
    repo, revision = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float32).to(DEVICE).eval()
    return tok, model


def chat_texts(tok, prompts, system=None):
    """Chat-formatted prompts. system=None keeps the template's default; system="" gives an empty system message."""
    def messages(p):
        return ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": p}]
    return [tok.apply_chat_template(messages(p), tokenize=False, add_generation_prompt=True) for p in prompts]


class PresencePenalty(LogitsProcessor):
    """vLLM/OpenAI presence penalty: subtract `penalty` from the logit of every token already in the generated text."""

    def __init__(self, penalty, prompt_len):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids, scores):
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, input_ids[:, self.prompt_len:], True)
        return scores - self.penalty * seen


def generate_batch(model, tok, prompts, system=None, max_new_tokens=MAX_NEW_TOKENS, sampling=None, seed=0,
                   greedy_repetition_penalty=1.0):
    """One batch. Greedy unless `sampling` is given. Greedy is pure (repetition_penalty=1.0, overriding a repo
    default) unless greedy_repetition_penalty=None, which keeps the repo's generation_config (used by the pilot).
    Returns (reply, n_new_tokens, hit_token_limit) per prompt."""
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    batch = tok(chat_texts(tok, prompts, system), return_tensors="pt", padding=True).to(DEVICE)
    prompt_len = batch["input_ids"].shape[1]
    kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=stop, pad_token_id=tok.pad_token_id, do_sample=False)
    if greedy_repetition_penalty is not None:
        kwargs["repetition_penalty"] = greedy_repetition_penalty
    if sampling:
        settings = dict(sampling)
        penalty = settings.pop("presence_penalty")
        kwargs.update(do_sample=True, **settings)
        if penalty:
            kwargs["logits_processor"] = [PresencePenalty(penalty, prompt_len)]
        torch.manual_seed(seed)
    with torch.no_grad():
        generated = model.generate(**batch, **kwargs)[:, prompt_len:].cpu()
    out = []
    for row in generated:
        ends = [i for i, t in enumerate(row.tolist()) if t in stop]
        n = ends[0] + 1 if ends else len(row)
        out.append((tok.decode(row[:n], skip_special_tokens=True).strip(), n, not ends))
    return out


def language_detector(restricted):
    builder = (LanguageDetectorBuilder.from_languages(*LANG_CODES) if restricted
               else LanguageDetectorBuilder.from_all_languages())
    return builder.with_preloaded_language_models().build()


def language_of(detector, text):
    lang = detector.detect_language_of(text)
    return "none" if lang is None else LANG_CODES.get(lang, "other")


def half_languages(detector, text):
    """Labels of the reply's first and second halves; a reply too short to split gets its whole label twice."""
    words = text.split()
    if len(words) < MIN_WORDS_TO_SPLIT:
        whole = language_of(detector, text)
        return whole, whole
    mid = len(words) // 2
    return language_of(detector, " ".join(words[:mid])), language_of(detector, " ".join(words[mid:]))


def in_language(row, lang):
    return row["lang_first"] == lang and row["lang_second"] == lang


def failure_language(row, lang):
    """The first half's label if it is not `lang`, else the second half's."""
    return row["lang_first"] if row["lang_first"] != lang else row["lang_second"]


def fixed_by_language(base_rows, rows):
    """Base failures (two-halves rule, Swedish prompts) by failure language, and how many of them `rows` has in Swedish."""
    other = {r["id"]: r for r in rows}
    out = {}
    for r in base_rows:
        if not in_language(r, "sv"):
            counts = out.setdefault(failure_language(r, "sv"), {"base_failures": 0, "fixed": 0})
            counts["base_failures"] += 1
            counts["fixed"] += in_language(other[r["id"]], "sv")
    return out


def last_token_residuals(model, tok, prompts):
    """Residual stream after every decoder layer at the last prompt token: tensor [n_prompts, n_layers, d_model].
    Runs the decoder stack only: full-vocabulary logits for every position of a batch of paragraphs need gigabytes."""
    layers = model.model.layers
    store = {}

    def hook(idx):
        def fn(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store[idx] = hidden[:, -1, :].float().cpu()
        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    out = []
    for start in range(0, len(prompts), BATCH):
        batch = tok(chat_texts(tok, prompts[start:start + BATCH]), return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            model.model(**batch)
        out.append(torch.stack([store[i] for i in range(len(layers))], dim=1))
    for handle in handles:
        handle.remove()
    return torch.cat(out)


def writers(model, layers):
    """Matrices that write into the residual stream in `layers`: attention output and MLP down projection."""
    out = []
    for i in layers:
        layer = model.model.layers[i]
        attn_out = layer.self_attn.o_proj if hasattr(layer, "self_attn") else layer.linear_attn.out_proj
        out += [(f"layers.{i}.attn_out", attn_out.weight), (f"layers.{i}.mlp.down_proj", layer.mlp.down_proj.weight)]
    return out


def scale_component(model, direction, layers, alpha, target_norms=None):
    """W <- W - a * r r^T W for every residual writer in `layers`: the component of what these matrices write along
    the unit direction r is multiplied by (1 - a). a = 1 removes it (Arditi et al. 2024, eq. 5, on a subset of
    layers), a = 2 flips its sign, a = -1 doubles it, a = -2 triples it. r and -r give the identical edit.

    a = alpha, or, when `target_norms` is given, the per-matrix value with the sign of alpha that makes
    ||a r r^T W||_F equal the target. Returns the saved originals and, per matrix, the Frobenius norms of the change
    and of the original matrix, the strength used and the number of entries that changed."""
    r = (direction / direction.norm()).to(DEVICE, torch.float32)
    saved, info = {}, {}
    with torch.no_grad():
        for name, W in writers(model, layers):
            rW = r @ W
            a = alpha if target_norms is None else math.copysign(target_norms[name] / float(rW.norm()), alpha)
            saved[name] = (W, W.detach().clone())
            W -= a * torch.outer(r, rW)
            info[name] = {"frobenius": float((a * rW).norm()), "frobenius_original": float(saved[name][1].norm()),
                          "strength": a,
                          "entries_changed": int((W != saved[name][1]).sum()), "entries": W.numel()}
    return saved, info


def restore(saved):
    with torch.no_grad():
        for W, original in saved.values():
            W.copy_(original)


def token_logprob_sums(model, tok, prefixes, continuations):
    """Sum of log-probabilities of each continuation given its prefix (prefix tokens are not scored), and its length.
    An empty prefix scores every token after the first. Sequences are scored one at a time, and the MPS cache is
    emptied after each, to bound memory."""
    out = []
    for prefix, cont in zip(prefixes, continuations):
        p = tok(prefix, add_special_tokens=False)["input_ids"]
        c = tok(cont, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + c], device=DEVICE)
        start = max(len(p), 1)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, start - 1:-1].float()
        logp = torch.log_softmax(logits, -1).gather(1, ids[0, start:, None]).sum()
        out.append((float(logp), ids.shape[1] - start))
        torch.mps.empty_cache()
    return out
