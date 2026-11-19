"""Experiment steps. Usage: python experiment.py STEP [--model qwen25|qwen35] [--quick] [--resume]

  langid   check the language detector on labelled Wikipedia paragraphs (sv, no, da, en)
  pilot    base model, greedy with the repo's default generation settings, first 40 Swedish dev prompts, 128 tokens
  dev128   dev round 1 (128 tokens, whole-reply language label): base run; fewer than 20 failures, so the plan's
           fallback direction (Swedish minus English text); removed (alpha 1) or flipped (alpha 2) in 4 layer windows
  base512  base model on all Swedish dev prompts at 512 tokens, compared with the 128-token run
  dev512   dev round 2 (512 tokens, two-halves language rule): 8 settings, selection rule, English check, diagnostics
  lora     LoRA baseline trained on dev prompts only (replies the base model gives under the Swedish system prompt)
  test     every condition on the test sets, once

--quick: 10 prompts per set from the dev files only (it never reads a test file); outputs go to results_quick/.
--resume: reuse generation files that already hold every item (after an interruption).
"""
import argparse
import collections
import json
import math
import random
import time

import torch

from common import (BATCH, DEVICE, MAX_NEW_TOKENS, MODELS, QA_MAX_NEW_TOKENS, ROOT, SAMPLING, SYSTEM_SV,
                    append_jsonl, chat_texts, failure_language, fixed_by_language, generate_batch, half_languages,
                    in_language, language_detector, language_of, last_token_residuals, load, read_jsonl, restore,
                    scale_component, token_logprob_sums)

DATA = ROOT / "data"
LANGS = ["sv", "no", "da", "en"]
QUICK_N = 10
ROUND1_MAX_NEW_TOKENS = 128
ROUND1_WINDOWS = [(4, 7), (8, 11), (12, 15), (16, 19)]
ROUND1_ALPHAS = [1.0, 2.0]
ROUND2_WINDOWS = [(12, 15), (16, 19)]
# svEn: the round-1 Swedish-minus-English direction, its component multiplied (alpha -1: x2, alpha -2: x3).
# fail: wrong-language minus Swedish replies of the 512-token base run, removed (alpha 1) or flipped (alpha 2).
ROUND2_SETTINGS = [("svEn", -1.0), ("svEn", -2.0), ("fail", 1.0), ("fail", 2.0)]
MIN_FAILURES = 20
MAX_PPL_RATIO = 1.15
MAX_ENGLISH_CHANGE = 0.05
RANDOM_SEEDS = [1, 2, 3, 4, 5]
SAMPLING_SEEDS = [1, 2, 3]
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0, epochs=3, lr=2e-4, batch=8, seed=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
RESUME = False


def run_generation(model, tok, items, path, detector, key="prompt", system=None, max_new_tokens=MAX_NEW_TOKENS,
                   sampling=None, seed=0, greedy_repetition_penalty=1.0):
    """Generate for every item and append one row per item to `path` as each batch finishes.
    With --resume, a file that already holds every item is reused and the time is reported as None."""
    if RESUME and path.exists():
        rows = read_jsonl(path)
        if [r["id"] for r in rows] == [it["id"] for it in items]:
            print("reusing", path.name, flush=True)
            return rows, None
    path.unlink(missing_ok=True)
    rows = []
    t0 = time.time()
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        outs = generate_batch(model, tok, [it[key] for it in chunk], system, max_new_tokens, sampling,
                              seed * 100003 + start, greedy_repetition_penalty)
        for it, (reply, n_tokens, truncated) in zip(chunk, outs):
            first, second = half_languages(detector, reply)
            row = {"id": it["id"], key: it[key], "reply": reply, "lang": language_of(detector, reply),
                   "lang_first": first, "lang_second": second, "n_tokens": n_tokens, "hit_token_limit": truncated}
            append_jsonl(path, row)
            rows.append(row)
        torch.mps.empty_cache()
    return rows, time.time() - t0


def langid(out):
    items = read_jsonl(DATA / "langid_check.jsonl")
    rows = [{"id": it["id"], "lang": it["lang"], "title": it["title"]} for it in items]
    summary = {}
    for restricted in [False, True]:
        detector = language_detector(restricted)
        name = "restricted_sv_no_da_en" if restricted else "all_languages"
        for field in ["text", "first_sentence"]:
            preds = [language_of(detector, it[field]) for it in items]
            for row, pred in zip(rows, preds):
                row[f"{name}/{field}"] = pred
            confusion = collections.Counter((it["lang"], p) for it, p in zip(items, preds))
            sv_pred = [p == "sv" for p in preds]
            sv_true = [it["lang"] == "sv" for it in items]
            summary[f"{name}/{field}"] = {
                "accuracy": sum(it["lang"] == p for it, p in zip(items, preds)) / len(items),
                "per_language_accuracy": {lang: sum(p == lang for it, p in zip(items, preds) if it["lang"] == lang)
                                          / sum(it["lang"] == lang for it in items) for lang in LANGS},
                "swedish_recall": sum(t and p for t, p in zip(sv_true, sv_pred)) / sum(sv_true),
                "swedish_precision": sum(t and p for t, p in zip(sv_true, sv_pred)) / max(sum(sv_pred), 1),
                "confusion_true_to_pred": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
                "n": len(items),
                "mean_chars": sum(len(it[field]) for it in items) / len(items),
            }
    with open(out / "langid_check.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    json.dump(summary, open(out / "langid_check_summary.json", "w"), indent=1)
    for key, s in summary.items():
        print(key, round(s["accuracy"], 3), s["per_language_accuracy"])


def pilot(model_key, n, out):
    tok, model = load(model_key)
    detector = language_detector(restricted=False)
    items = read_jsonl(DATA / "sv_dev.jsonl")[:n]
    torch.mps.synchronize()
    rows, seconds = run_generation(model, tok, items, out / f"pilot_{model_key}.jsonl", detector,
                                   max_new_tokens=ROUND1_MAX_NEW_TOKENS, greedy_repetition_penalty=None)
    langs = [r["lang"] for r in rows]
    tokens = sum(r["n_tokens"] for r in rows)
    summary = {"model": model_key, "revision": MODELS[model_key][1], "n_prompts": len(rows),
               "decoding": "greedy with the repo's default generation_config (qwen25: repetition_penalty=1.1)",
               "max_new_tokens": ROUND1_MAX_NEW_TOKENS, "languages": dict(collections.Counter(langs)),
               "wrong_language_rate": sum(lang != "sv" for lang in langs) / len(langs),
               "seconds": seconds, "generated_tokens": tokens, "tokens_per_second": tokens / seconds,
               "hit_token_limit": sum(r["hit_token_limit"] for r in rows)}
    json.dump(summary, open(out / f"pilot_{model_key}_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def is_swedish_whole(row):
    return row["lang"] == "sv"


def is_swedish_halves(row):
    return in_language(row, "sv")


def swedish_reply_ppl(model, tok, prompts, rows, is_swedish):
    """Perplexity, under `model`, of the replies classified as Swedish, each conditioned on its default-template prompt."""
    pairs = [(p, r["reply"]) for p, r in zip(chat_texts(tok, prompts), rows) if is_swedish(r)]
    if not pairs:
        return math.inf
    sums = token_logprob_sums(model, tok, [p for p, _ in pairs], [c for _, c in pairs])
    return math.exp(-sum(s for s, _ in sums) / sum(n for _, n in sums))


def edit_size(info, model):
    entries = sum(v["entries"] for v in info.values())
    changed = sum(v["entries_changed"] for v in info.values())
    total = sum(p.numel() for p in model.parameters())
    return {"free_parameters": model.config.hidden_size + 1,
            "free_parameters_are": f"one {model.config.hidden_size}-number direction and one strength "
                                   "(plus the discrete choice of layer window)",
            "frobenius_norm_of_change": math.sqrt(sum(v["frobenius"] ** 2 for v in info.values())),
            "frobenius_norm_of_edited_matrices": math.sqrt(sum(v["frobenius_original"] ** 2 for v in info.values())),
            "matrices": len(info), "entries_in_edited_matrices": entries, "entries_changed": changed,
            "total_parameters": total, "percent_of_weights_changed": 100 * changed / total}


def evaluate_setting(model, tok, items, path, detector, direction, layers, alpha, max_new_tokens=MAX_NEW_TOKENS):
    saved, info = scale_component(model, direction, layers, alpha)
    rows, seconds = run_generation(model, tok, items, path, detector, max_new_tokens=max_new_tokens)
    restore(saved)
    return rows, seconds, edit_size(info, model)


def dev128(model_key, out, quick):
    """Round 1, as run: 128 tokens, whole-reply language label."""
    tok, model = load(model_key)
    detector = language_detector(restricted=False)
    items = read_jsonl(DATA / "sv_dev.jsonl")[:QUICK_N if quick else None]
    prompts = [it["prompt"] for it in items]
    base_rows, base_seconds = run_generation(model, tok, items, out / "dev_base.jsonl", detector,
                                             max_new_tokens=ROUND1_MAX_NEW_TOKENS)
    wrong = torch.tensor([not is_swedish_whole(r) for r in base_rows])
    acts = last_token_residuals(model, tok, prompts)
    if int(wrong.sum()) >= MIN_FAILURES:
        source = "mean(last prompt token | wrong-language reply) - mean(last prompt token | Swedish reply), dev"
        pos, neg = acts[wrong], acts[~wrong]
    else:
        source = ("fallback (fewer than 20 failures): mean(last prompt token | Swedish Wikipedia paragraph as prompt)"
                  " - mean(... | English paragraph), data/langid_check.jsonl")
        paras = read_jsonl(DATA / "langid_check.jsonl")
        pos = last_token_residuals(model, tok, [p["text"] for p in paras if p["lang"] == "sv"])
        neg = last_token_residuals(model, tok, [p["text"] for p in paras if p["lang"] == "en"])
    directions = {(lo, hi): pos[:, hi].mean(0) - neg[:, hi].mean(0) for lo, hi in ROUND1_WINDOWS}
    torch.save({f"{lo}-{hi}": d for (lo, hi), d in directions.items()}, out / "dev_directions.pt")

    combos = []
    for (lo, hi) in ROUND1_WINDOWS:
        for alpha in ROUND1_ALPHAS:
            layers = list(range(lo, hi + 1))
            name = f"L{lo}-{hi}_a{alpha:g}"
            rows, seconds, size = evaluate_setting(model, tok, items, out / f"dev_edit_{name}.jsonl", detector,
                                                   directions[(lo, hi)], layers, alpha, ROUND1_MAX_NEW_TOKENS)
            combos.append({"name": name, "layers": layers, "alpha": alpha, "rows": rows, "seconds": seconds,
                           "size": size})

    base_ppl = swedish_reply_ppl(model, tok, prompts, base_rows, is_swedish_whole)
    for c in combos:
        c["swedish_rate"] = sum(is_swedish_whole(r) for r in c["rows"]) / len(c["rows"])
        c["swedish_reply_ppl_under_base"] = swedish_reply_ppl(model, tok, prompts, c["rows"], is_swedish_whole)
        c["ppl_ratio_to_base"] = c["swedish_reply_ppl_under_base"] / base_ppl
        c["eligible"] = c["ppl_ratio_to_base"] <= MAX_PPL_RATIO
    eligible = [c for c in combos if c["eligible"]]
    best = max(eligible, key=lambda c: (c["swedish_rate"], -c["alpha"], -c["layers"][0])) if eligible else None
    summary = {
        "model": model_key, "revision": MODELS[model_key][1], "n_dev": len(items),
        "max_new_tokens": ROUND1_MAX_NEW_TOKENS, "language_rule": "whole reply classified Swedish",
        "base": {"swedish_rate": 1 - float(wrong.float().mean()), "n_wrong": int(wrong.sum()),
                 "languages": dict(collections.Counter(r["lang"] for r in base_rows)),
                 "swedish_reply_ppl_under_base": base_ppl, "seconds": base_seconds},
        "direction_source": source,
        "direction_layer": "residual stream after the last layer of each window, at the last prompt token",
        "direction_raw_norms": {f"{lo}-{hi}": float(d.norm()) for (lo, hi), d in directions.items()},
        "selection_rule": f"highest dev Swedish-reply rate among settings whose Swedish dev replies have perplexity "
                          f"under the base model <= {MAX_PPL_RATIO} x the base model's own; ties: smaller alpha, "
                          f"then earlier window",
        "combos": [{k: v for k, v in c.items() if k != "rows"} for c in combos],
        "selected": None if best is None else {"name": best["name"], "layers": best["layers"], "alpha": best["alpha"],
                                               "direction_window": f"{best['layers'][0]}-{best['layers'][-1]}"},
        "feedback_budget": {"dev_generation_runs_of_base_model": 1, "activation_passes_over_dev_prompts": 1,
                            "dev_evaluations_of_edited_models": len(combos),
                            "dev_generations_total": len(items) * (1 + len(combos))},
    }
    json.dump(summary, open(out / "dev_selection.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "combos"}, indent=1))


def base512(model_key, out, quick):
    """Item 3: the base model at 512 tokens; failures vs the 128-token run, drift, and the first-128-token check."""
    tok, model = load(model_key)
    detector = language_detector(restricted=False)
    items = read_jsonl(DATA / "sv_dev.jsonl")[:QUICK_N if quick else None]
    rows, seconds = run_generation(model, tok, items, out / "dev512_base.jsonl", detector)
    short = {r["id"]: r for r in read_jsonl(out / "dev_base.jsonl")}
    short_halves = {r["id"]: half_languages(detector, short[r["id"]]["reply"]) for r in rows}
    fail512 = [r["id"] for r in rows if not is_swedish_halves(r)]
    fail128_whole = [r["id"] for r in rows if not is_swedish_whole(short[r["id"]])]
    fail128_halves = [i for i, (a, b) in short_halves.items() if not a == b == "sv"]
    prefix_mismatch = []
    for r in rows:
        s = short[r["id"]]
        same = (r["reply"].startswith(s["reply"].rstrip("�")) if s["hit_token_limit"]
                else r["reply"] == s["reply"])
        if not same:
            prefix_mismatch.append(r["id"])
    drift = collections.Counter(f"{r['lang_first']}->{r['lang_second']}" for r in rows
                                if r["lang_first"] != r["lang_second"])
    tokens = sum(r["n_tokens"] for r in rows)
    summary = {
        "model": model_key, "revision": MODELS[model_key][1], "n_dev": len(rows), "max_new_tokens": MAX_NEW_TOKENS,
        "language_rule": "a reply is Swedish only if both halves are classified Swedish (common.half_languages)",
        "failures_512_two_halves": {"n": len(fail512), "ids": fail512,
                                    "by_language": dict(collections.Counter(failure_language(r, "sv") for r in rows
                                                                            if not is_swedish_halves(r)))},
        "failures_128_whole_reply_as_run": {"n": len(fail128_whole), "ids": fail128_whole},
        "failures_128_two_halves_relabelled": {"n": len(fail128_halves), "ids": fail128_halves},
        "failing_at_both_512_and_128_whole": sorted(set(fail512) & set(fail128_whole)),
        "drift_first_half_to_second_half": {"n": sum(drift.values()), "by_pair": dict(drift)},
        "hit_token_limit": sum(r["hit_token_limit"] for r in rows),
        "first_128_tokens_match_dev_base": {
            "rule": "128-token reply is a prefix of the 512-token reply (equal if it ended before 128 tokens)",
            "n_checked": len(rows), "n_mismatch": len(prefix_mismatch), "mismatch_ids": prefix_mismatch},
        "seconds": seconds, "generated_tokens": tokens,
        "tokens_per_second": None if seconds is None else tokens / seconds,
    }
    json.dump(summary, open(out / "dev512_base_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "model"}, indent=1))


def failure_direction_diagnostics(acts, wrong, sv_en):
    """Per window: split-half cosine of the failure direction (200 random splits), leave-one-out AUC of the projection
    (direction re-estimated without the held-out prompt), and its cosine with the Swedish-minus-English direction."""
    g = torch.Generator().manual_seed(0)
    out = {"n_wrong": int(wrong.sum()), "n_right": int((~wrong).sum()), "windows": {}}
    for lo, hi in ROUND1_WINDOWS:
        pos, neg = acts[wrong][:, hi], acts[~wrong][:, hi]
        d = pos.mean(0) - neg.mean(0)
        cos = []
        for _ in range(200):
            pp, nn = torch.randperm(len(pos), generator=g), torch.randperm(len(neg), generator=g)
            d1 = pos[pp[:len(pp) // 2]].mean(0) - neg[nn[:len(nn) // 2]].mean(0)
            d2 = pos[pp[len(pp) // 2:]].mean(0) - neg[nn[len(nn) // 2:]].mean(0)
            cos.append(float(torch.nn.functional.cosine_similarity(d1, d2, dim=0)))
        loo = []
        for i in range(len(acts)):
            keep = torch.ones(len(acts), dtype=torch.bool)
            keep[i] = False
            di = acts[keep & wrong][:, hi].mean(0) - acts[keep & ~wrong][:, hi].mean(0)
            loo.append(float(acts[i, hi] @ (di / di.norm())))
        loo = torch.tensor(loo)
        cos = sorted(cos)
        out["windows"][f"{lo}-{hi}"] = {
            "split_half_cosine_median": cos[len(cos) // 2], "split_half_cosine_p05": cos[int(0.05 * len(cos))],
            "leave_one_out_auc_wrong_vs_right": float((loo[wrong][:, None] > loo[~wrong][None, :]).float().mean()),
            "cosine_with_sv_minus_en": float(torch.nn.functional.cosine_similarity(d, sv_en[f"{lo}-{hi}"], dim=0)),
            "raw_norm": float(d.norm())}
    return out


def language_positions(para_acts, sv_en):
    """Projection of each language's paragraphs (last prompt token) on the unit Swedish-minus-English direction,
    and the position of each language's mean on a scale where English is 0 and Swedish is 1."""
    out = {"note": "sv and en are the paragraphs the direction was computed from; no and da are new to it",
           "windows": {}}
    for lo, hi in ROUND1_WINDOWS:
        r = sv_en[f"{lo}-{hi}"] / sv_en[f"{lo}-{hi}"].norm()
        proj = {lang: para_acts[lang][:, hi] @ r for lang in LANGS}
        m_sv, m_en = float(proj["sv"].mean()), float(proj["en"].mean())
        out["windows"][f"{lo}-{hi}"] = {lang: {"n": len(p), "mean": float(p.mean()), "sd": float(p.std()),
                                               "position_en0_sv1": (float(p.mean()) - m_en) / (m_sv - m_en)}
                                        for lang, p in proj.items()}
    return out


def dev512(model_key, out, quick):
    """Round 2: 512 tokens, two-halves rule. 8 settings, selection rule, English check, diagnostics."""
    tok, model = load(model_key)
    detector = language_detector(restricted=False)
    n = QUICK_N if quick else None
    items = read_jsonl(DATA / "sv_dev.jsonl")[:n]
    en_items = read_jsonl(DATA / "en_dev.jsonl")[:n]
    prompts = [it["prompt"] for it in items]
    base_rows = read_jsonl(out / "dev512_base.jsonl")
    assert [r["id"] for r in base_rows] == [it["id"] for it in items]
    wrong = torch.tensor([not is_swedish_halves(r) for r in base_rows])
    if not 0 < int(wrong.sum()) < len(items):
        raise SystemExit("the failure direction needs at least one failure and one Swedish reply")
    acts = last_token_residuals(model, tok, prompts)
    sv_en = torch.load(out / "dev_directions.pt")
    paras = read_jsonl(DATA / "langid_check.jsonl")
    para_acts = {lang: last_token_residuals(model, tok, [p["text"] for p in paras if p["lang"] == lang])
                 for lang in LANGS}
    short_wrong = torch.tensor([not is_swedish_whole(r) for r in read_jsonl(out / "dev_base.jsonl")[:len(items)]])
    diagnostics = {
        "failure_direction": {
            "512_tokens_two_halves_rule": failure_direction_diagnostics(acts, wrong, sv_en),
            "128_tokens_whole_reply_rule": failure_direction_diagnostics(acts, short_wrong, sv_en)},
        "language_positions_on_sv_minus_en": language_positions(para_acts, sv_en)}
    json.dump(diagnostics, open(out / "dev512_diagnostics.json", "w"), indent=1)

    directions = {}
    for lo, hi in ROUND2_WINDOWS:
        directions[f"svEn/{lo}-{hi}"] = sv_en[f"{lo}-{hi}"]
        directions[f"fail/{lo}-{hi}"] = acts[wrong][:, hi].mean(0) - acts[~wrong][:, hi].mean(0)
    torch.save(directions, out / "dev512_directions.pt")

    settings = []
    for lo, hi in ROUND2_WINDOWS:
        for source, alpha in ROUND2_SETTINGS:
            s = {"name": f"{source}_L{lo}-{hi}_a{alpha:g}", "direction": f"{source}/{lo}-{hi}",
                 "layers": list(range(lo, hi + 1)), "alpha": alpha}
            rows, s["seconds"], s["size"] = evaluate_setting(model, tok, items, out / f"dev512_edit_{s['name']}.jsonl",
                                                             detector, directions[s["direction"]], s["layers"], alpha)
            s["rows"] = rows
            settings.append(s)
            print(s["name"], "Swedish rate", sum(is_swedish_halves(r) for r in rows) / len(rows), flush=True)

    base_ppl = swedish_reply_ppl(model, tok, prompts, base_rows, is_swedish_halves)
    for s in settings:
        s["swedish_rate"] = sum(is_swedish_halves(r) for r in s["rows"]) / len(s["rows"])
        s["failure_languages"] = dict(collections.Counter(failure_language(r, "sv") for r in s["rows"]
                                                          if not is_swedish_halves(r)))
        s["base_failures_fixed_by_language"] = fixed_by_language(base_rows, s["rows"])
        s["swedish_reply_ppl_under_base"] = swedish_reply_ppl(model, tok, prompts, s["rows"], is_swedish_halves)
        s["ppl_ratio_to_base"] = s["swedish_reply_ppl_under_base"] / base_ppl
        s["eligible"] = s["ppl_ratio_to_base"] <= MAX_PPL_RATIO
    ranked = sorted([s for s in settings if s["eligible"]],
                    key=lambda s: (-s["swedish_rate"], abs(s["alpha"]), s["layers"][0],
                                   s["size"]["frobenius_norm_of_change"]))
    if not ranked:
        raise SystemExit("no setting passes the perplexity rule")

    en_base, en_base_seconds = run_generation(model, tok, en_items, out / "dev512_en_base.jsonl", detector)
    en_base_rate = sum(in_language(r, "en") for r in en_base) / len(en_base)
    selected, english_checks = None, []
    for s in ranked:
        saved, _ = scale_component(model, directions[s["direction"]], s["layers"], s["alpha"])
        rows, _ = run_generation(model, tok, en_items, out / f"dev512_en_{s['name']}.jsonl", detector)
        restore(saved)
        rate = sum(in_language(r, "en") for r in rows) / len(rows)
        english_checks.append({"name": s["name"], "english_rate": rate,
                               "passed": abs(rate - en_base_rate) <= MAX_ENGLISH_CHANGE})
        print(english_checks[-1], flush=True)
        if english_checks[-1]["passed"]:
            selected = s
            break
    english_check_passed = selected is not None
    selected = selected or ranked[0]

    round1 = json.load(open(out / "dev_selection.json"))["feedback_budget"]["dev_evaluations_of_edited_models"]
    summary = {
        "model": model_key, "revision": MODELS[model_key][1], "n_dev": len(items), "n_en_dev": len(en_items),
        "max_new_tokens": MAX_NEW_TOKENS,
        "language_rule": "a reply is in language X only if both halves are classified X (common.half_languages)",
        "base": {"swedish_rate": sum(is_swedish_halves(r) for r in base_rows) / len(base_rows),
                 "n_wrong": int(wrong.sum()), "swedish_reply_ppl_under_base": base_ppl,
                 "english_rate_on_english_dev": en_base_rate, "english_dev_seconds": en_base_seconds},
        "min_failures_rule": {"required": MIN_FAILURES, "failures": int(wrong.sum()),
                              "relaxed": int(wrong.sum()) < MIN_FAILURES},
        "directions": {"svEn": "round-1 Swedish-minus-English direction (results/dev_directions.pt), component "
                               "multiplied: alpha -1 doubles it, alpha -2 triples it",
                       "fail": "mean(last prompt token | wrong-language reply at 512 tokens) - mean(... | Swedish "
                               "reply), dev; alpha 1 removes it, alpha 2 flips it",
                       "layer": "residual stream after the last layer of each window, at the last prompt token",
                       "raw_norms": {k: float(v.norm()) for k, v in directions.items()}},
        "selection_rule": f"eligible if the Swedish dev replies' perplexity under the base model is <= {MAX_PPL_RATIO} "
                          "x the base model's own; rank by dev Swedish-reply rate, ties by smaller |alpha|, earlier "
                          "window, smaller Frobenius norm of the change; then, in that order, take the first setting "
                          f"whose English-reply rate on the English dev prompts is within {MAX_ENGLISH_CHANGE:.0%} "
                          "(absolute) of the base model's",
        "settings": [{k: v for k, v in s.items() if k != "rows"} for s in settings],
        "ranking": [s["name"] for s in ranked],
        "english_checks": english_checks,
        "selected": {"name": selected["name"], "direction": selected["direction"], "layers": selected["layers"],
                     "alpha": selected["alpha"], "english_check_passed": english_check_passed},
        "feedback_budget": {
            "round1_dev_evaluations_of_edited_models": round1,
            "round2_swedish_dev_evaluations_of_edited_models": len(settings),
            "round2_english_dev_evaluations_of_edited_models": len(english_checks),
            "total_dev_evaluations_of_edited_models": round1 + len(settings) + len(english_checks),
            "base_model_dev_runs": {"swedish_128_tokens": 1, "swedish_512_tokens": 1, "english_512_tokens": 1},
            "activation_passes_over_dev_prompts": 2,
            "dev_generations_512_tokens": len(items) * (1 + len(settings)) + len(en_items) * (1 + len(english_checks))},
    }
    json.dump(summary, open(out / "dev512_selection.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ["base", "ranking", "english_checks", "selected", "feedback_budget"]},
                     indent=1))
    for s in summary["settings"]:
        print(s["name"], "sv_rate", round(s["swedish_rate"], 3), "ppl_ratio", round(s["ppl_ratio_to_base"], 3),
              "eligible", s["eligible"], "frobenius", round(s["size"]["frobenius_norm_of_change"], 2))


def lora_train(model_key, out, quick, dev_eval):
    from peft import LoraConfig, get_peft_model
    tok, model = load(model_key)
    detector = language_detector(restricted=False)
    items = read_jsonl(DATA / "sv_dev.jsonl")[:QUICK_N if quick else None]
    rows, _ = run_generation(model, tok, items, out / "lora_targets.jsonl", detector, system=SYSTEM_SV)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    examples = []
    for it, r in zip(items, rows):
        if not is_swedish_halves(r):
            continue
        prompt_ids = tok(chat_texts(tok, [it["prompt"]])[0], add_special_tokens=False)["input_ids"]
        reply_ids = tok(r["reply"], add_special_tokens=False)["input_ids"] + ([] if r["hit_token_limit"] else [end])
        examples.append((prompt_ids, reply_ids))
    torch.manual_seed(LORA["seed"])
    config = LoraConfig(r=LORA["r"], lora_alpha=LORA["lora_alpha"], lora_dropout=LORA["lora_dropout"],
                        target_modules=LORA["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LORA["lr"])
    rng = random.Random(LORA["seed"])
    log = []
    epochs = 1 if quick else LORA["epochs"]
    t0 = time.time()
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), LORA["batch"]):
            chunk = [examples[i] for i in order[start:start + LORA["batch"]]]
            n_target = sum(len(c) for _, c in chunk)
            batch_loss = 0.0
            for p, c in chunk:
                ids = torch.tensor([p + c], device=DEVICE)
                logits = model(input_ids=ids).logits[0, len(p) - 1:-1]
                loss = torch.nn.functional.cross_entropy(logits, ids[0, len(p):], reduction="sum") / n_target
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            log.append({"epoch": epoch, "step": len(log), "loss": batch_loss})
            print(log[-1], flush=True)
    train_seconds = time.time() - t0
    model.save_pretrained(out / "lora_adapter")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = model.merge_and_unload().eval()
    summary = {"model": model_key, "revision": MODELS[model_key][1], "settings": LORA, "epochs_run": epochs,
               "batching": f"{LORA['batch']} sequences per optimizer step, run one at a time with gradients summed; "
                           "loss = mean over all reply tokens in the step",
               "n_dev_prompts": len(items), "n_training_examples": len(examples), "max_new_tokens": MAX_NEW_TOKENS,
               "targets": "the base model's own greedy replies to dev prompts under the system prompt "
                          f"'{SYSTEM_SV}', kept if Swedish by the two-halves rule; trained without that system prompt",
               "trainable_parameters": trainable, "train_seconds": train_seconds, "loss_log": log,
               "dev_evaluation_run": dev_eval, "dev_swedish_rate_after_training": None,
               "feedback_budget": {"dev_generation_runs_for_targets": 1,
                                   "dev_evaluations_of_trained_model": int(dev_eval)}}
    if dev_eval:
        dev_rows, _ = run_generation(model, tok, items, out / "dev512_lora.jsonl", detector)
        summary["dev_swedish_rate_after_training"] = sum(is_swedish_halves(r) for r in dev_rows) / len(dev_rows)
    json.dump(summary, open(out / "lora_summary.json", "w"), indent=1)
    print({k: summary.get(k) for k in ["n_training_examples", "train_seconds", "dev_swedish_rate_after_training"]})


def test(model_key, out, quick):
    from peft import PeftModel
    tok, model = load(model_key)
    detector = language_detector(restricted=False)
    if quick:
        paras = read_jsonl(DATA / "langid_check.jsonl")
        sets = {"sv": read_jsonl(DATA / "sv_dev.jsonl")[:QUICK_N], "en": read_jsonl(DATA / "en_dev.jsonl")[:QUICK_N]}
        wiki = ([p for p in paras if p["lang"] == "sv"][:QUICK_N // 2]
                + [p for p in paras if p["lang"] == "en"][:QUICK_N // 2])
        inputs = {"sv": "data/sv_dev.jsonl", "en": "data/en_dev.jsonl", "wiki": "data/langid_check.jsonl (sv, en)"}
    else:
        sets = {"sv": read_jsonl(DATA / "sv_test.jsonl"), "en": read_jsonl(DATA / "en_test.jsonl"),
                "qa": read_jsonl(DATA / "qa_test.jsonl")}
        wiki = read_jsonl(DATA / "wiki_ppl.jsonl")
        inputs = {"sv": "data/sv_test.jsonl", "en": "data/en_test.jsonl", "qa": "data/qa_test.jsonl",
                  "wiki": "data/wiki_ppl.jsonl"}
    max_new = {"sv": MAX_NEW_TOKENS, "en": MAX_NEW_TOKENS, "qa": QA_MAX_NEW_TOKENS}
    selection = json.load(open(out / "dev512_selection.json"))["selected"]
    direction = torch.load(out / "dev512_directions.pt")[selection["direction"]]
    layers, alpha = selection["layers"], selection["alpha"]
    timings, sizes = {}, {}

    def run_condition(name, **kwargs):
        t0 = time.time()
        for set_name, items in sets.items():
            key = "question" if set_name == "qa" else "prompt"
            run_generation(model, tok, items, out / f"test_{name}_{set_name}.jsonl", detector, key=key,
                           max_new_tokens=max_new[set_name], **kwargs)
        timings[name] = time.time() - t0
        print(name, round(timings[name], 1), "s", flush=True)

    def wiki_nll(name):
        path = out / f"test_wiki_nll_{name}.jsonl"
        path.unlink(missing_ok=True)
        for p, (logp, n_tok) in zip(wiki, token_logprob_sums(model, tok, [""] * len(wiki), [w["text"] for w in wiki])):
            append_jsonl(path, {"id": p["id"], "lang": p["lang"], "logprob_sum": logp, "n_tokens": n_tok})

    run_condition("base")
    wiki_nll("base")
    saved, info = scale_component(model, direction, layers, alpha)
    sizes["edited"] = edit_size(info, model)
    run_condition("edited")
    wiki_nll("edited")
    restore(saved)
    target_norms = {name: v["frobenius"] for name, v in info.items()}
    for seed in RANDOM_SEEDS:
        u = torch.randn(direction.numel(), generator=torch.Generator().manual_seed(seed))
        saved, info_r = scale_component(model, u, layers, alpha, target_norms=target_norms)
        sizes[f"random_s{seed}"] = {**edit_size(info_r, model), "cosine_with_edit_direction":
                                    float(torch.nn.functional.cosine_similarity(u, direction, dim=0)),
                                    "per_matrix_strength": {k: v["strength"] for k, v in info_r.items()}}
        run_condition(f"random_s{seed}")
        wiki_nll(f"random_s{seed}")
        restore(saved)
    run_condition("sysprompt_sv", system=SYSTEM_SV)
    run_condition("sysprompt_empty", system="")
    for seed in SAMPLING_SEEDS:
        run_condition(f"sampling_s{seed}", sampling=SAMPLING[model_key], seed=seed)
    model = PeftModel.from_pretrained(model, out / "lora_adapter").merge_and_unload().eval()
    run_condition("lora")
    wiki_nll("lora")

    tok, model = load(model_key)
    path = out / "test_reply_nll_under_base.jsonl"
    path.unlink(missing_ok=True)
    prefixes = dict(zip([it["id"] for it in sets["sv"]], chat_texts(tok, [it["prompt"] for it in sets["sv"]])))
    for name in timings:
        rows = [r for r in read_jsonl(out / f"test_{name}_sv.jsonl") if is_swedish_halves(r)]
        sums = token_logprob_sums(model, tok, [prefixes[r["id"]] for r in rows], [r["reply"] for r in rows])
        for r, (logp, n_tok) in zip(rows, sums):
            append_jsonl(path, {"condition": name, "id": r["id"], "logprob_sum": logp, "n_tokens": n_tok})

    config = {"model": model_key, "repo": MODELS[model_key][0], "revision": MODELS[model_key][1],
              "device": "mps", "dtype": "float32", "batch_size": BATCH, "inputs": inputs, "max_new_tokens": max_new,
              "language_rule": "a reply is in language X only if both halves are classified X (common.half_languages)",
              "greedy": "do_sample=False, repetition_penalty=1.0", "chat_template": "model default; non-thinking",
              "system_prompts": {"base/edited/random/lora/sampling": "template default",
                                 "sysprompt_sv": SYSTEM_SV, "sysprompt_empty": ""},
              "sampling": SAMPLING[model_key], "sampling_seeds": SAMPLING_SEEDS,
              "edit": selection, "random_direction_seeds": RANDOM_SEEDS,
              "random_control": "standard-normal direction, unit norm, same layers and sign of alpha; per-matrix "
                                "strength set so the Frobenius norm of each matrix change equals the real edit's",
              "edit_sizes": sizes, "seconds_per_condition": timings, "lora": LORA}
    json.dump(config, open(out / "test_config.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["langid", "pilot", "dev128", "base512", "dev512", "lora", "test"])
    ap.add_argument("--model", default="qwen25", choices=list(MODELS))
    ap.add_argument("--quick", action="store_true", help="10 dev prompts per set, outputs to results_quick/")
    ap.add_argument("--resume", action="store_true", help="reuse complete generation files")
    ap.add_argument("--skip-dev-eval", action="store_true", help="lora: skip the trained model's dev evaluation")
    args = ap.parse_args()
    RESUME = args.resume
    out = ROOT / ("results_quick" if args.quick else "results")
    out.mkdir(exist_ok=True)
    if args.step == "langid":
        langid(out)
    elif args.step == "pilot":
        pilot(args.model, QUICK_N if args.quick else 40, out)
    elif args.step == "dev128":
        dev128(args.model, out, args.quick)
    elif args.step == "base512":
        base512(args.model, out, args.quick)
    elif args.step == "dev512":
        dev512(args.model, out, args.quick)
    elif args.step == "lora":
        lora_train(args.model, out, args.quick, dev_eval=not args.skip_dev_eval)
    elif args.step == "test":
        test(args.model, out, args.quick)
