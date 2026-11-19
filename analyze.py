"""Metrics, paired statistics, examples, the understanding-annotation sheet and the results tables in NOTE.md.

Usage: python analyze.py [--quick]          summary.json, summary.md, examples.md
       python analyze.py sheet [--quick]    understanding_sheet.jsonl (blinded; labels go in understanding_labels.jsonl)
       python analyze.py note               fills every <!-- table:NAME --> ... <!-- /table:NAME --> block in NOTE.md

The main analysis leaves out the first 10 items of every test file: the --quick run read them (and its summary
showed every condition's results on them) before --quick was changed to use dev files only. "full" uses all items.
"""
import argparse
import collections
import hashlib
import json
import math
import random
import re

import numpy as np
from scipy.stats import binomtest

from common import ROOT, failure_language, fixed_by_language, in_language, read_jsonl

DATA = ROOT / "data"
RANDOM = [f"random_s{i}" for i in range(1, 6)]
SAMPLING = [f"sampling_s{i}" for i in range(1, 4)]
CONDITIONS = ["base", "edited"] + RANDOM + ["sysprompt_sv", "sysprompt_empty"] + SAMPLING + ["lora"]
GROUPS = {"random": RANDOM, "sampling": SAMPLING}
VS_EDITED = ["base", "random", "sysprompt_sv", "sysprompt_empty", "sampling", "lora"] + RANDOM
KEY_COMPARISONS = [("edited", "base"), ("edited", "random"), ("sysprompt_sv", "base"), ("edited", "sysprompt_sv"),
                   ("sysprompt_empty", "base"), ("sampling", "base"), ("lora", "base"), ("edited", "lora")]
QUICK_TOUCHED = 10
TEST_FILES = ["sv_test", "en_test", "qa_test", "wiki_ppl"]
BOOTSTRAP = 10000
FLAG_PPL_RATIO = 1.15
LABELS = ["yes", "no", "unclear"]
DESCRIPTIONS = {
    "overview": "This table shows the main result per condition: how often it replied in Swedish to Swedish prompts, "
                "how many base failures it fixed and how many new ones it caused, and two regression checks.",
    "key comparisons": "Each cell is a paired difference in percentage points on the same prompts, with a 95% "
                       "bootstrap interval and, for yes/no outcomes, the exact McNemar p-value.",
    "churn, edit vs random edits": "This table compares how many Swedish prompts the edit changed with how many each "
                                   "random edit of the same size changed.",
    "paired comparisons": "This table lists every paired difference, against base and against the edited model.",
    "base failures fixed, by language": "This table shows, per condition, how many of the base model's wrong-language "
                                        "replies became Swedish, split by the language of the base reply.",
    "perplexity checks": "Perplexity is how surprised a model is by a text (lower is better). The first table scores "
                         "each condition's Swedish replies with the base model. The second scores Wikipedia "
                         "paragraphs with each set of weights.",
    "power": "This line says how large a difference the test could reliably detect.",
    "understanding and label check": "This table shows, for each source's wrong-language replies, whether lingua's "
                                     "label was right and whether the reply understood the prompt (hand labels, see "
                                     "understanding_rubric.md).",
    "base failures by label and understanding": "This table splits the base model's failures by the two hand labels.",
    "English prompts newly broken": "This table lists English prompts that got an English reply from the base model "
                                    "but not from the condition, with lingua's labels for the two halves.",
    "conditions": "This table shows every condition separately, including each random edit and sampling seed. Drift "
                  "means the two halves of a reply are in different languages.",
    "dev round 1 (128 tokens)": "This table shows round 1 of the dev work: the plan's eight settings at 128 tokens.",
    "dev base run, 512 vs 128 tokens": "This table compares the base model's dev run at 512 tokens with the 128-token run.",
    "dev round 2 (512 tokens)": "This table shows round 2 of the dev work: eight settings at 512 tokens, the "
                                "selection and the English check.",
    "failure-direction diagnostics": "This table shows how stable the wrong-minus-Swedish direction is. The cosine "
                                     "compares directions from two random halves of the data (1 means identical). "
                                     "The AUC says how well it separates failures from Swedish replies on held-out "
                                     "prompts (0.5 is chance).",
    "Norwegian and Danish on the Swedish-minus-English direction": "This table shows where each language's Wikipedia "
                                                                   "paragraphs fall on the Swedish-minus-English "
                                                                   "direction.",
    "edit size": "This table shows the size of the selected edit and of the random edits. Free parameters are the "
                 "numbers that define an edit.",
    "pilot": "This table shows the pilot on 40 dev prompts that chose the model.",
    "LoRA": "LoRA is a small add-on to the weights, trained on examples.",
}


def qa_correct(reply, answers):
    return any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", reply, re.IGNORECASE) for a in answers)


def touched_ids():
    return {it["id"] for f in TEST_FILES for it in read_jsonl(DATA / f"{f}.jsonl")[:QUICK_TOUCHED]}


def load_condition(out, name, keep, qa_items):
    sets = ["sv", "en", "qa"] if qa_items else ["sv", "en"]
    rows = {s: [r for r in read_jsonl(out / f"test_{name}_{s}.jsonl") if keep(r["id"])] for s in sets}
    d = {"rows": rows, "sv": np.array([in_language(r, "sv") for r in rows["sv"]], float),
         "en": np.array([in_language(r, "en") for r in rows["en"]], float)}
    if qa_items:
        d["qa"] = np.array([qa_correct(r["reply"], qa_items[r["id"]]["answers"]) for r in rows["qa"]], float)
        d["qa_lang"] = np.array([qa_items[r["id"]]["lang"] for r in rows["qa"]])
    return d


def load_subset(out, keep, qa_items):
    data = {n: load_condition(out, n, keep, qa_items) for n in CONDITIONS}
    for g, names in GROUPS.items():
        data[g] = {k: np.mean([data[n][k] for n in names], axis=0) for k in metrics(qa_items)}
    return data


def metrics(qa_items):
    return ["sv", "en", "qa"] if qa_items else ["sv", "en"]


def bootstrap_ci(x, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (BOOTSTRAP, len(x)))
    diffs = (x[idx] - y[idx]).mean(1)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def paired(x, y):
    """x - y on the same prompts: mean difference, paired-bootstrap 95% CI, and an exact McNemar test when binary."""
    out = {"a": float(x.mean()), "b": float(y.mean()), "diff": float((x - y).mean()), "ci95": bootstrap_ci(x, y)}
    if set(np.unique(np.concatenate([x, y]))) <= {0.0, 1.0}:
        gain, loss = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        out.update(gained=gain, lost=loss,
                   mcnemar_p=1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue))
    return out


def mde(discordant_rate, n, z=1.959964 + 0.841621):
    """Smallest paired difference detectable with 80% power at alpha=0.05 (two-sided), normal approximation."""
    return z * math.sqrt(discordant_rate / n) / math.sqrt(1 + z * z / n)


def wilson(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(method="wilson")
    return [float(ci.low), float(ci.high)]


def reply_ppl(out, keep):
    """Base-model perplexity of each condition's Swedish replies, overall and on prompts where base also replied in Swedish."""
    by = {}
    for r in read_jsonl(out / "test_reply_nll_under_base.jsonl"):
        if keep(r["id"]):
            by.setdefault(r["condition"], {})[r["id"]] = (r["logprob_sum"], r["n_tokens"])
    base = by["base"]
    result = {}
    for name in CONDITIONS:
        cond = by.get(name, {})
        common = sorted(set(cond) & set(base))
        ppl_all = math.exp(-sum(v[0] for v in cond.values()) / sum(v[1] for v in cond.values())) if cond else None
        ratio = math.exp(-sum(cond[i][0] for i in common) / sum(cond[i][1] for i in common)
                         + sum(base[i][0] for i in common) / sum(base[i][1] for i in common)) if common else None
        x = np.array([-cond[i][0] / cond[i][1] for i in common])
        y = np.array([-base[i][0] / base[i][1] for i in common])
        result[name] = {"n_swedish_replies": len(cond), "ppl_all_swedish_replies": ppl_all,
                        "n_common_with_base": len(common), "ppl_ratio_to_base_on_common": ratio,
                        "nll_per_token_diff_vs_base": paired(x, y) if common else None,
                        "flag_clearly_worse": bool(ratio is not None and ratio > FLAG_PPL_RATIO)}
    return result


def wiki_ppl(out, keep):
    result = {}
    base = {r["id"]: r for r in read_jsonl(out / "test_wiki_nll_base.jsonl")}
    for name in ["base", "edited"] + RANDOM + ["lora"]:
        rows = [r for r in read_jsonl(out / f"test_wiki_nll_{name}.jsonl") if keep(r["id"])]
        result[name] = {}
        for lang in ["sv", "en"]:
            sel = [r for r in rows if r["lang"] == lang]
            x = np.array([-r["logprob_sum"] / r["n_tokens"] for r in sel])
            y = np.array([-base[r["id"]]["logprob_sum"] / base[r["id"]]["n_tokens"] for r in sel])
            result[name][lang] = {"n": len(sel),
                                  "ppl": math.exp(-sum(r["logprob_sum"] for r in sel) / sum(r["n_tokens"] for r in sel)),
                                  "nll_per_token_diff_vs_base": paired(x, y)}
    return result


def item_id(row):
    return hashlib.sha1((row["id"] + "\x00" + row["reply"]).encode()).hexdigest()[:10]


def count(labels, field):
    return {k: sum(lab[field] == k for lab in labels) for k in LABELS}


def understanding(out, data, labels):
    """Annotation counts per source: did the reply understand the prompt, and is the language label right."""
    sources = {"dev_base_128": [r for r in read_jsonl(out / "dev_base.jsonl") if r["lang"] != "sv"],
               "dev_base_512": [r for r in read_jsonl(out / "dev512_base.jsonl") if not in_language(r, "sv")]}
    for n in CONDITIONS:
        sources[f"test_{n}"] = [r for r in data[n]["rows"]["sv"] if not in_language(r, "sv")]
    result = {}
    for name, rows in sources.items():
        labs = [labels[item_id(r)] for r in rows]
        result[name] = {"wrong_language_replies": len(labs), "label_ok": count(labs, "label_ok"),
                        "understood": count(labs, "understood"),
                        "label_ok_by_understood": {f"{a}/{b}": sum(lab["label_ok"] == a and lab["understood"] == b
                                                                    for lab in labs) for a in LABELS for b in LABELS},
                        "real_switch_and_understood": sum(lab["label_ok"] == "yes" and lab["understood"] == "yes"
                                                          for lab in labs)}
    base_sv = {r["id"]: r for r in data["base"]["rows"]["sv"]}
    for n in CONDITIONS[1:]:
        rescued = [r for r in data[n]["rows"]["sv"]
                   if in_language(r, "sv") and not in_language(base_sv[r["id"]], "sv")]
        labs = [labels[item_id(r)] for r in rescued]
        result[f"test_{n}"]["rescued"] = {"n": len(labs), "label_ok": count(labs, "label_ok"),
                                          "understood": count(labs, "understood")}
    return result


def sheet(out):
    """Blinded sheet: every non-Swedish reply to a Swedish prompt (both dev base runs and all test conditions) and
    every test reply that is Swedish where the base reply was not, deduplicated, shuffled, without condition labels."""
    data = {n: load_condition(out, n, lambda i: True, None) for n in CONDITIONS}
    base_sv = {r["id"]: r for r in data["base"]["rows"]["sv"]}
    rows = [r for r in read_jsonl(out / "dev_base.jsonl") if r["lang"] != "sv"]
    rows += [r for r in read_jsonl(out / "dev512_base.jsonl") if not in_language(r, "sv")]
    for n in CONDITIONS:
        rows += [r for r in data[n]["rows"]["sv"]
                 if not in_language(r, "sv") or not in_language(base_sv[r["id"]], "sv")]
    items = {}
    for r in rows:
        lingua = f"{r['lang_first']} / {r['lang_second']}" if "lang_first" in r else f"{r['lang']} (whole reply)"
        items[item_id(r)] = {"item": item_id(r), "prompt": r["prompt"], "reply": r["reply"], "lingua": lingua}
    items = list(items.values())
    random.Random(0).shuffle(items)
    with open(out / "understanding_sheet.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(len(items), "items")


def examples(out, data, k=10):
    base, edited, sysp = (data[n]["rows"]["sv"] for n in ["base", "edited", "sysprompt_sv"])
    order = sorted(range(len(base)), key=lambda i: (in_language(base[i], "sv"),
                                                     in_language(edited[i], "sv") == in_language(base[i], "sv"),
                                                     base[i]["id"]))
    lines = ["# Side-by-side examples (Swedish test prompts, main subset)", "",
             "Selection rule (fixed in code): prompts where the base reply was not Swedish first, then prompts where "
             "the edited model's Swedish/not-Swedish outcome differs from base, then by prompt id. Labels are "
             "lingua's for the first / second half of the reply.", ""]
    for i in order[:k]:
        lines += [f"## {base[i]['id']}: {base[i]['prompt']}", ""]
        for label, rows in [("Base", base), ("Edited", edited), ("System prompt", sysp)]:
            reply = rows[i]["reply"].replace("\n", " / ")
            lines += [f"**{label}** [{rows[i]['lang_first']} / {rows[i]['lang_second']}]: {reply}", ""]
    (out / "examples.md").write_text("\n".join(lines), encoding="utf-8")


def condition_stats(d, qa):
    rows = d["rows"]["sv"]
    n = len(rows)
    s = {"n_sv": n, "swedish_rate": float(d["sv"].mean()), "swedish_rate_wilson95": wilson(d["sv"].sum(), n),
         "failure_languages": dict(collections.Counter(failure_language(r, "sv") for r in rows
                                                       if not in_language(r, "sv"))),
         "drift_first_to_second_half": sum(r["lang_first"] != r["lang_second"] for r in rows),
         "sv_hit_token_limit": sum(r["hit_token_limit"] for r in rows) / n,
         "n_en": len(d["en"]), "english_rate_on_english_prompts": float(d["en"].mean())}
    if qa:
        s.update(n_qa=len(d["qa"]), qa_accuracy=float(d["qa"].mean()),
                 qa_accuracy_sv=float(d["qa"][d["qa_lang"] == "sv"].mean()),
                 qa_accuracy_en=float(d["qa"][d["qa_lang"] == "en"].mean()))
    return s


def subset_results(out, data, qa_items, keep, labels):
    ms = metrics(qa_items)
    n_sv = len(data["base"]["sv"])
    res = {"conditions": {n: condition_stats(data[n], qa_items) for n in CONDITIONS}}
    for g, names in GROUPS.items():
        res["conditions"][f"{g}_mean"] = {f"mean_{k}": float(data[g][k].mean()) for k in ms}
    res["paired_vs_base"] = {n: {k: paired(data[n][k], data["base"][k]) for k in ms}
                             for n in CONDITIONS[1:] + list(GROUPS)}
    res["paired_vs_edited"] = {n: {k: paired(data["edited"][k], data[n][k]) for k in ms} for n in VS_EDITED}
    p = res["paired_vs_base"]["edited"]["sv"]
    disc = (p["gained"] + p["lost"]) / n_sv
    res["power"] = {"n": n_sv, "observed_discordant_rate_edited_vs_base": disc,
                    "mde_80pct_power_at_observed_discordance": mde(max(disc, 1 / n_sv), n_sv),
                    "mde_by_discordant_rate": {f"{r:.2f}": mde(r, n_sv) for r in [0.05, 0.10, 0.20, 0.30]},
                    "base_failures_available_to_fix": int(n_sv - data["base"]["sv"].sum())}
    res["base_failures_fixed_by_language"] = {n: fixed_by_language(data["base"]["rows"]["sv"], data[n]["rows"]["sv"])
                                              for n in CONDITIONS[1:]}
    base_en = {r["id"]: in_language(r, "en") for r in data["base"]["rows"]["en"]}
    res["english_newly_broken"] = {n: [{"id": r["id"], "lang_first": r["lang_first"], "lang_second": r["lang_second"]}
                                       for r in data[n]["rows"]["en"] if base_en[r["id"]] and not in_language(r, "en")]
                                   for n in CONDITIONS[1:]}
    res["reply_ppl_under_base"] = reply_ppl(out, keep)
    res["wikipedia_ppl"] = wiki_ppl(out, keep)
    res["understanding"] = understanding(out, data, labels) if labels else None
    return res


def summarize(out, quick):
    qa_items = None if quick else {q["id"]: q for q in read_jsonl(DATA / "qa_test.jsonl")}
    touched = set() if quick else touched_ids()
    labels_path = out / "understanding_labels.jsonl"
    labels = {r["item"]: r for r in read_jsonl(labels_path)} if labels_path.exists() else None
    subsets = {"main": lambda i: i not in touched}
    if not quick:
        subsets["full"] = lambda i: True
    summary = {"subsets": {}, "main_subset": f"all test items except the first {QUICK_TOUCHED} of each test file, "
                                             "which the --quick run read before it was changed (none in --quick)"}
    for name, keep in subsets.items():
        data = load_subset(out, keep, qa_items)
        summary["subsets"][name] = subset_results(out, data, qa_items, keep, labels)
        if name == "main":
            examples(out, data)
    summary["pilot"] = {}
    for model_key in ["qwen35", "qwen25"]:
        p = json.load(open(out / f"pilot_{model_key}_summary.json"))
        wrong = round(p["wrong_language_rate"] * p["n_prompts"])
        summary["pilot"][model_key] = {**p, "n_wrong": wrong, "wrong_rate_wilson95": wilson(wrong, p["n_prompts"])}
    for key, name in [("dev128", "dev_selection"), ("base512", "dev512_base_summary"), ("dev512", "dev512_selection"),
                      ("diagnostics", "dev512_diagnostics"), ("lora", "lora_summary"), ("test_config", "test_config")]:
        summary[key] = json.load(open(out / f"{name}.json"))
    base512 = {r["id"]: in_language(r, "sv") for r in read_jsonl(out / "dev512_base.jsonl")}
    summary["dev512_base_swedish_wilson95"] = wilson(sum(base512.values()), len(base512))
    summary["dev512_churn"] = {}
    for c in summary["dev512"]["settings"]:
        rows = read_jsonl(out / f"dev512_edit_{c['name']}.jsonl")
        summary["dev512_churn"][c["name"]] = {
            "fixed": sum(not base512[r["id"]] and in_language(r, "sv") for r in rows),
            "newly_broken": sum(base512[r["id"]] and not in_language(r, "sv") for r in rows)}
    summary["dev128_reply_languages"] = {
        c["name"]: dict(collections.Counter(r["lang"] for r in read_jsonl(out / f"dev_edit_{c['name']}.jsonl")))
        for c in summary["dev128"]["combos"]}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    t = tables(summary)
    (out / "summary.md").write_text("# Results summary (generated by analyze.py from results/)\n\n" + "\n\n".join(
        f"## {name}\n\n{DESCRIPTIONS[name.split(': ', 1)[-1]]}\n\n" + "\n".join(lines) for name, lines in t.items())
        + "\n", encoding="utf-8")


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def triple(c):
    return " / ".join(str(c[k]) for k in LABELS)


def churn(res, name, metric):
    """Prompts fixed / newly broken relative to base (for a group: mean over its members)."""
    names = GROUPS.get(name, [name])
    fixed = np.mean([res["paired_vs_base"][n][metric]["gained"] for n in names])
    broken = np.mean([res["paired_vs_base"][n][metric]["lost"] for n in names])
    return f"{fixed:.1f} / {broken:.1f}" if name in GROUPS else f"{fixed:.0f} / {broken:.0f}"


def condition_table(res):
    lines = ["| Condition | Swedish replies to Swedish prompts [95% CI] | fixed / newly broken vs base | English "
             "replies to English prompts | fixed / newly broken vs base | QA accuracy | Swedish replies that drift "
             "(first half vs second) | Swedish prompts hitting 512 tokens |", "|---|---|---|---|---|---|---|---|"]
    for name in CONDITIONS:
        c = res["conditions"][name]
        lo, hi = c["swedish_rate_wilson95"]
        sv, en = ("", "") if name == "base" else (churn(res, name, "sv"), churn(res, name, "en"))
        lines.append(f"| {name} | {pct(c['swedish_rate'])} [{pct(lo)}, {pct(hi)}] (n={c['n_sv']}) | {sv} | "
                     f"{pct(c['english_rate_on_english_prompts'])} (n={c['n_en']}) | {en} | {pct(c.get('qa_accuracy'))} "
                     f"| {c['drift_first_to_second_half']} | {pct(c['sv_hit_token_limit'])} |")
    for g, names in GROUPS.items():
        r = res["conditions"][f"{g}_mean"]
        lines.append(f"| {g}, mean of {len(names)} | {pct(r['mean_sv'])} | {churn(res, g, 'sv')} | {pct(r['mean_en'])} "
                     f"| {churn(res, g, 'en')} | {pct(r.get('mean_qa'))} | | |")
    return lines


def overview_table(res):
    """The main conditions in one table; random edits and sampling as means over their draws."""
    lines = ["| Condition | Swedish replies to Swedish prompts [95% CI] | fixed / newly broken vs base | "
             "English replies to English prompts | QA accuracy |", "|---|---|---|---|---|"]
    for name in ["base", "edited", "random", "sysprompt_sv", "sysprompt_empty", "sampling", "lora"]:
        if name in GROUPS:
            r = res["conditions"][f"{name}_mean"]
            label, sv, en, qa = (f"{name}, mean of {len(GROUPS[name])}", pct(r["mean_sv"]), pct(r["mean_en"]),
                                 pct(r.get("mean_qa")))
        else:
            c = res["conditions"][name]
            lo, hi = c["swedish_rate_wilson95"]
            label, sv, en, qa = (name, f"{pct(c['swedish_rate'])} [{pct(lo)}, {pct(hi)}]",
                                 pct(c["english_rate_on_english_prompts"]), pct(c.get("qa_accuracy")))
        lines.append(f"| {label} | {sv} | {'' if name == 'base' else churn(res, name, 'sv')} | {en} | {qa} |")
    return lines


def base_failure_table(res):
    """The base model's wrong-language replies, by whether lingua was right and whether the reply understood the prompt."""
    x = res["understanding"]["test_base"]["label_ok_by_understood"]
    rows = [("yes", "real wrong-language reply"), ("no", "detector error (the reply is Swedish)"),
            ("unclear", "unclear label")]
    lines = ["| Base failure is a | understood the prompt | misread it | unclear | total |", "|---|---|---|---|---|"]
    for key, name in rows:
        cells = [x[f"{key}/{u}"] for u in LABELS]
        lines.append(f"| {name} | " + " | ".join(str(c) for c in cells) + f" | {sum(cells)} |")
    return lines


def broken_english_table(res):
    lines = ["| Condition | English prompt | lingua: first half / second half |", "|---|---|---|"]
    for name, rows in res["english_newly_broken"].items():
        lines += [f"| {name} | {r['id']} | {r['lang_first']} / {r['lang_second']} |" for r in rows]
    return lines


def churn_table(res):
    """The edit's churn next to each random edit of the same size (Swedish prompts, relative to base)."""
    lines = ["| Weights | fixed | newly broken | changed (fixed + broken) | net |", "|---|---|---|---|---|"]
    for name in ["edited"] + RANDOM:
        p = res["paired_vs_base"][name]["sv"]
        lines.append(f"| {name} | {p['gained']} | {p['lost']} | {p['gained'] + p['lost']} | {p['gained'] - p['lost']:+d} |")
    return lines


def paired_table(res):
    lines = ["| Comparison | Metric | Difference | 95% CI | gained / lost | McNemar p |", "|---|---|---|---|---|---|"]
    for title, key in [("{} minus base", "paired_vs_base"), ("edited minus {}", "paired_vs_edited")]:
        for name, ms in res[key].items():
            for metric, p in ms.items():
                extra = f"{p['gained']} / {p['lost']} | {p['mcnemar_p']:.3g}" if "mcnemar_p" in p else " | "
                lines.append(f"| {title.format(name)} | {metric} | {100 * p['diff']:+.1f} pts | "
                             f"[{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}] | {extra} |")
    return lines


def key_table(res):
    """The main paired comparisons: difference in points, paired-bootstrap 95% CI and, for binary metrics, McNemar p."""
    ms = list(res["paired_vs_base"]["edited"])
    names = {"sv": "Swedish replies (Swedish prompts)", "en": "English replies (English prompts)", "qa": "QA accuracy"}
    lines = ["| Comparison | " + " | ".join(names[m] for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for a, b in KEY_COMPARISONS:
        ps = res["paired_vs_base"][a] if b == "base" else res["paired_vs_edited"][b]
        cells = [f"{100 * p['diff']:+.1f} [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}]"
                 + (f", p={p['mcnemar_p']:.2g}" if "mcnemar_p" in p else "") for p in (ps[m] for m in ms)]
        lines.append(f"| {a} minus {b} | " + " | ".join(cells) + " |")
    return lines


def fixed_table(res):
    langs = sorted({lang for f in res["base_failures_fixed_by_language"].values() for lang in f})
    lines = ["| Condition | " + " | ".join(f"base failures in '{lang}' now Swedish" for lang in langs) + " |",
             "|---|" + "---|" * len(langs)]
    for name, f in res["base_failures_fixed_by_language"].items():
        lines.append(f"| {name} | " + " | ".join(f"{f[lang]['fixed']} of {f[lang]['base_failures']}" if lang in f
                                                 else "" for lang in langs) + " |")
    return lines


def ppl_table(res):
    lines = ["| Condition | n Swedish replies | ppl under base (all) | ratio to base on prompts both answered in "
             "Swedish | flag (>1.15) |", "|---|---|---|---|---|"]
    for name, r in res["reply_ppl_under_base"].items():
        if r["ppl_all_swedish_replies"] is not None:
            lines.append(f"| {name} | {r['n_swedish_replies']} | {r['ppl_all_swedish_replies']:.3f} | "
                         f"{r['ppl_ratio_to_base_on_common']:.3f} (n={r['n_common_with_base']}) | "
                         f"{r['flag_clearly_worse']} |")
    lines += ["", "| Weights | Swedish Wikipedia ppl | English Wikipedia ppl |", "|---|---|---|"]
    for name, r in res["wikipedia_ppl"].items():
        lines.append(f"| {name} | {r['sv']['ppl']:.3f} (n={r['sv']['n']}) | {r['en']['ppl']:.3f} (n={r['en']['n']}) |")
    return lines


def understanding_table(res):
    u = res["understanding"]
    if u is None:
        return ["Not annotated yet."]
    lines = ["| Source | non-Swedish replies | label right? yes / no / unclear | understood the prompt? yes / no / "
             "unclear | real switch and understood | rescued Swedish replies: n; understood yes / no / unclear |",
             "|---|---|---|---|---|---|"]
    for name, x in u.items():
        r = x.get("rescued")
        resc = f"{r['n']}; {triple(r['understood'])}" if r else ""
        lines.append(f"| {name} | {x['wrong_language_replies']} | {triple(x['label_ok'])} | "
                     f"{triple(x['understood'])} | {x['real_switch_and_understood']} | {resc} |")
    return lines


def dev_tables(s):
    d1, b, d2 = s["dev128"], s["base512"], s["dev512"]
    round1 = [f"Base: {pct(d1['base']['swedish_rate'])} Swedish ({d1['base']['n_wrong']} failures of {d1['n_dev']}); "
              f"direction: {d1['direction_source']}.", "",
              "| Setting | Swedish rate | reply languages | ppl ratio to base | eligible | Frobenius norm of change |",
              "|---|---|---|---|---|---|"]
    for c in d1["combos"]:
        round1.append(f"| {c['name']} | {pct(c['swedish_rate'])} | {s['dev128_reply_languages'][c['name']]} | "
                      f"{c['ppl_ratio_to_base']:.3f} | {c['eligible']} | {c['size']['frobenius_norm_of_change']:.2f} |")
    f512, f128, f128h = (b["failures_512_two_halves"], b["failures_128_whole_reply_as_run"],
                         b["failures_128_two_halves_relabelled"])
    base = ["| Measure | Value |", "|---|---|",
            f"| failures at 128 tokens, whole-reply label (as run) | {f128['n']} of {b['n_dev']} |",
            f"| failures at 128 tokens, two-halves rule | {f128h['n']} of {b['n_dev']} |",
            f"| failures at 512 tokens, two-halves rule | {f512['n']} of {b['n_dev']} ({f512['by_language']}) |",
            f"| failing both at 512 and at 128 (whole) | {len(b['failing_at_both_512_and_128_whole'])} |",
            f"| replies whose halves differ in language (drift) at 512 | {b['drift_first_half_to_second_half']['n']} "
            f"({b['drift_first_half_to_second_half']['by_pair']}) |",
            f"| replies hitting 512 tokens | {b['hit_token_limit']} |",
            f"| first 128 tokens differ from results/dev_base.jsonl | {b['first_128_tokens_match_dev_base']['n_mismatch']} "
            f"{b['first_128_tokens_match_dev_base']['mismatch_ids']} |"]
    english = {c["name"]: c for c in d2["english_checks"]}
    lo, hi = s["dev512_base_swedish_wilson95"]
    round2 = [f"Base: {pct(d2['base']['swedish_rate'])} Swedish [{pct(lo)}, {pct(hi)}] ({d2['base']['n_wrong']} failures "
              f"of {d2['n_dev']}); "
              f"English dev: {pct(d2['base']['english_rate_on_english_dev'])} English (n={d2['n_en_dev']}). "
              f"Minimum-failures rule relaxed: {d2['min_failures_rule']['relaxed']}.", "",
              "| Setting | Swedish rate | fixed / newly broken vs base | failure languages | ppl ratio to base | "
              "eligible | Frobenius norm of change | English rate on English dev (check) |",
              "|---|---|---|---|---|---|---|---|"]
    for c in d2["settings"]:
        e = english.get(c["name"])
        check = f"{pct(e['english_rate'])} ({'pass' if e['passed'] else 'fail'})" if e else "not checked"
        ch = s["dev512_churn"][c["name"]]
        round2.append(f"| {c['name']} | {pct(c['swedish_rate'])} | {ch['fixed']} / {ch['newly_broken']} | "
                      f"{c['failure_languages']} | "
                      f"{c['ppl_ratio_to_base']:.3f} | {c['eligible']} | {c['size']['frobenius_norm_of_change']:.2f} | "
                      f"{check} |")
    sel, fb = d2["selected"], d2["feedback_budget"]
    round2 += ["", f"Selected: {sel['name']} (English check passed: {sel['english_check_passed']}). Feedback budget: "
                   f"{fb['total_dev_evaluations_of_edited_models']} dev evaluations of edited models "
                   f"({fb['round1_dev_evaluations_of_edited_models']} in round 1, "
                   f"{fb['round2_swedish_dev_evaluations_of_edited_models']} Swedish and "
                   f"{fb['round2_english_dev_evaluations_of_edited_models']} English in round 2)."]
    return {"dev round 1 (128 tokens)": round1, "dev base run, 512 vs 128 tokens": base,
            "dev round 2 (512 tokens)": round2}


def diagnostics_tables(s):
    diag = s["diagnostics"]
    fail = ["| Failure labels | Window | split-half cosine, median (5th pct) | leave-one-out AUC | cosine with "
            "Swedish-minus-English |", "|---|---|---|---|---|"]
    for labels, x in diag["failure_direction"].items():
        for w, v in x["windows"].items():
            fail.append(f"| {labels} ({x['n_wrong']} vs {x['n_right']}) | {w} | {v['split_half_cosine_median']:.2f} "
                        f"({v['split_half_cosine_p05']:.2f}) | {v['leave_one_out_auc_wrong_vs_right']:.2f} | "
                        f"{v['cosine_with_sv_minus_en']:.2f} |")
    pos = diag["language_positions_on_sv_minus_en"]
    langs = ["en", "da", "no", "sv"]
    positions = [pos["note"] + ". Position of each language's mean on a scale where English is 0 and Swedish is 1.", "",
                 "| Window | " + " | ".join(langs) + " |", "|---|" + "---|" * len(langs)]
    for w, v in pos["windows"].items():
        positions.append(f"| {w} | " + " | ".join(f"{v[lang]['position_en0_sv1']:.2f}" for lang in langs) + " |")
    return {"failure-direction diagnostics": fail, "Norwegian and Danish on the Swedish-minus-English direction": positions}


def size_table(s):
    lines = ["| Edit | free parameters | Frobenius norm of change | Frobenius norm of edited matrices | "
             "% of all weights changed |", "|---|---|---|---|---|"]
    for name, z in s["test_config"]["edit_sizes"].items():
        lines.append(f"| {name} | {z['free_parameters']} | {z['frobenius_norm_of_change']:.2f} | "
                     f"{z['frobenius_norm_of_edited_matrices']:.1f} | {z['percent_of_weights_changed']:.2f} |")
    return lines


def pilot_table(s):
    lines = ["Greedy decoding with each repo's default generation settings (for qwen25 this includes "
             "repetition_penalty=1.1; qwen35 ships none), default chat template, 128 new tokens, fp32 on MPS, "
             "first 40 Swedish dev prompts, whole-reply language label.", "",
             "| Model | Wrong language | Wilson 95% CI | tokens/s |", "|---|---|---|---|"]
    for model_key, p in s["pilot"].items():
        lo, hi = p["wrong_rate_wilson95"]
        lines.append(f"| {model_key} | {p['n_wrong']}/{p['n_prompts']} | [{pct(lo)}, {pct(hi)}] "
                     f"| {p['tokens_per_second']:.0f} |")
    return lines


def lora_table(s):
    lo = s["lora"]
    return [f"Training examples: {lo['n_training_examples']} of {lo['n_dev_prompts']} dev prompts; trainable "
            f"parameters: {lo['trainable_parameters']}; dev Swedish rate after training: "
            f"{pct(lo.get('dev_swedish_rate_after_training'))} (dev evaluation run: {lo['dev_evaluation_run']})."]


def tables(s):
    t = {}
    for name, res in s["subsets"].items():
        t[f"test {name}: conditions"] = condition_table(res)
        t[f"test {name}: overview"] = overview_table(res)
        t[f"test {name}: key comparisons"] = key_table(res)
        t[f"test {name}: churn, edit vs random edits"] = churn_table(res)
        t[f"test {name}: paired comparisons"] = paired_table(res)
        t[f"test {name}: base failures fixed, by language"] = fixed_table(res)
        t[f"test {name}: perplexity checks"] = ppl_table(res)
        pw = res["power"]
        t[f"test {name}: power"] = [
            f"n = {pw['n']}. Observed discordant rate (edited vs base): {pw['observed_discordant_rate_edited_vs_base']:.3f}. "
            f"Minimum detectable difference (80% power, two-sided 0.05): "
            f"{100 * pw['mde_80pct_power_at_observed_discordance']:.1f} pts. Base failures available to fix: "
            f"{pw['base_failures_available_to_fix']}."]
        t[f"test {name}: understanding and label check"] = understanding_table(res)
        if res["understanding"]:
            t[f"test {name}: base failures by label and understanding"] = base_failure_table(res)
        t[f"test {name}: English prompts newly broken"] = broken_english_table(res)
    t.update(dev_tables(s))
    t.update(diagnostics_tables(s))
    t["edit size"] = size_table(s)
    t["pilot"] = pilot_table(s)
    t["LoRA"] = lora_table(s)
    return t


APPENDIX = [
    ("0 of 40 wrong-language replies, Qwen3.5 pilot", "summary.json", "pilot.qwen35.n_wrong"),
    ("Wilson 95% CI 0-8.8%", "summary.json", "pilot.qwen35.wrong_rate_wilson95"),
    ("first 128 tokens (pilot)", "summary.json", "pilot.qwen35.max_new_tokens"),
    ("revision 7ae5576", "test_config.json", "revision"),
    ("lingua right on 100%", "langid_check_summary.json", "all_languages/text.accuracy"),
    ("of 200 paragraphs", "langid_check_summary.json", "all_languages/text.n"),
    ("up to 512 new tokens", "test_config.json", "max_new_tokens.sv"),
    ("one 896-number direction and one strength", "test_config.json", "edit_sizes.edited.free_parameters"),
    ("4.18% of all weights", "test_config.json", "edit_sizes.edited.percent_of_weights_changed"),
    ("8 matrices", "test_config.json", "edit_sizes.edited.matrices"),
    ("feedback budget 17", "dev512_selection.json", "feedback_budget.total_dev_evaluations_of_edited_models"),
    ("5 random directions", "test_config.json", "random_direction_seeds"),
    ("3 sampling seeds", "test_config.json", "sampling_seeds"),
    ("140 Swedish prompts", "summary.json", "subsets.main.conditions.base.n_sv"),
    ("90 English prompts", "summary.json", "subsets.main.conditions.base.n_en"),
    ("50 questions", "summary.json", "subsets.main.conditions.base.n_qa"),
    ("90 paragraphs (Swedish part)", "summary.json", "subsets.main.wikipedia_ppl.base.sv.n"),
    ("90 paragraphs (English part)", "summary.json", "subsets.main.wikipedia_ppl.base.en.n"),
    ("edit vs base: -2.9 points", "summary.json", "subsets.main.paired_vs_base.edited.sv.diff"),
    ("edit vs base: [-10.0, +3.6]", "summary.json", "subsets.main.paired_vs_base.edited.sv.ci95"),
    ("edit vs base: p = 0.54", "summary.json", "subsets.main.paired_vs_base.edited.sv.mcnemar_p"),
    ("edit vs random edits: -2.7", "summary.json", "subsets.main.paired_vs_edited.random.sv.diff"),
    ("edit fixed 10", "summary.json", "subsets.main.paired_vs_base.edited.sv.gained"),
    ("edit broke 14", "summary.json", "subsets.main.paired_vs_base.edited.sv.lost"),
    ("dev round 1: base (132 of 150 Swedish)", "dev_selection.json", "base.swedish_rate"),
    ("dev round 1: best setting L12-15_a1, one prompt more", "dev_selection.json", "combos.4.swedish_rate"),
    ("dev round 2: base", "dev512_selection.json", "base.swedish_rate"),
    ("dev round 2: selected setting fail_L12-15_a2, one prompt more", "dev512_selection.json", "settings.3.swedish_rate"),
    ("Swedish system prompt: +11.4", "summary.json", "subsets.main.paired_vs_base.sysprompt_sv.sv.diff"),
    ("Swedish system prompt: p = 0.007", "summary.json", "subsets.main.paired_vs_base.sysprompt_sv.sv.mcnemar_p"),
    ("it broke 6 English prompts", "summary.json", "subsets.main.paired_vs_base.sysprompt_sv.en.lost"),
    ("4 Swedish replies, en016, en096", "summary.json", "subsets.main.english_newly_broken.sysprompt_sv"),
    ("LoRA: +11.4", "summary.json", "subsets.main.paired_vs_base.lora.sv.diff"),
    ("LoRA: p = 0.011", "summary.json", "subsets.main.paired_vs_base.lora.sv.mcnemar_p"),
    ("Swedish Wikipedia perplexity, base: 47.2", "summary.json", "subsets.main.wikipedia_ppl.base.sv.ppl"),
    ("Swedish Wikipedia perplexity, LoRA: 123.5", "summary.json", "subsets.main.wikipedia_ppl.lora.sv.ppl"),
    ("Swedish Wikipedia perplexity, edit: 49.2", "summary.json", "subsets.main.wikipedia_ppl.edited.sv.ppl"),
    ("3 training epochs", "lora_summary.json", "epochs_run"),
    ("final training loss 0.0415", "lora_summary.json", "loss_log.53.loss"),
    ("LoRA QA gain: +10", "summary.json", "subsets.main.paired_vs_base.lora.qa.diff"),
    ("LoRA QA gain: p = 0.0625", "summary.json", "subsets.main.paired_vs_base.lora.qa.mcnemar_p"),
    ("empty system prompt: +6.4", "summary.json", "subsets.main.paired_vs_base.sysprompt_empty.sv.diff"),
    ("empty system prompt: p = 0.14", "summary.json", "subsets.main.paired_vs_base.sysprompt_empty.sv.mcnemar_p"),
    ("sampling: +5.0", "summary.json", "subsets.main.paired_vs_base.sampling.sv.diff"),
    ("32 base test failures", "summary.json", "subsets.main.understanding.test_base.wrong_language_replies"),
    ("23 real, 17 misread, 5 understood, 8 detector errors", "summary.json",
     "subsets.main.understanding.test_base.label_ok_by_understood"),
    ("English failures fixed by the Swedish system prompt: 13 of 16", "summary.json",
     "subsets.main.base_failures_fixed_by_language.sysprompt_sv.en"),
    ("English failures fixed by the empty system prompt: 10", "summary.json",
     "subsets.main.base_failures_fixed_by_language.sysprompt_empty.en"),
    ("English failures fixed by LoRA: 16", "summary.json", "subsets.main.base_failures_fixed_by_language.lora.en"),
    ("base on dev: 84.7%", "dev512_selection.json", "base.swedish_rate"),
    ("base on dev: 78.0-89.6%", "summary.json", "dev512_base_swedish_wilson95"),
    ("base on test: 77.1%", "summary.json", "subsets.main.conditions.base.swedish_rate"),
    ("base on test: 69.5-83.3%", "summary.json", "subsets.main.conditions.base.swedish_rate_wilson95"),
    ("detectable difference about 9.5 points", "summary.json", "subsets.main.power.mde_80pct_power_at_observed_discordance"),
]


def lookup(obj, path):
    for key in path.split("."):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def fmt_value(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt_value(x) for x in v) + "]"
    if isinstance(v, dict) and "id" in v:
        return f"{v['id']} {v['lang_first']}/{v['lang_second']}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {fmt_value(x)}" for k, x in v.items())
    return str(v)


def appendix_table(out):
    files = {}
    lines = ["| Number in the text | File | Field | Value in the file |", "|---|---|---|---|"]
    for text, name, path in APPENDIX:
        data = files.setdefault(name, json.load(open(out / name)))
        lines.append(f"| {text} | `results/{name}` | `{path}` | {fmt_value(lookup(data, path))} |")
    return lines


def fill_note(out):
    t = tables(json.load(open(out / "summary.json")))
    t["appendix"] = appendix_table(out)
    path = ROOT / "NOTE.md"
    text = path.read_text(encoding="utf-8")
    names = re.findall(r"<!-- table:(.+?) -->", text)
    for name in names:
        block = re.compile(rf"(<!-- table:{re.escape(name)} -->\n).*?(<!-- /table:{re.escape(name)} -->)", re.S)
        assert block.search(text), name
        text = block.sub(lambda m: m.group(1) + "\n".join(t[name]) + "\n" + m.group(2), text)
    path.write_text(text, encoding="utf-8")
    print("filled", len(names), "tables")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="summary", choices=["summary", "sheet", "note"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = ROOT / ("results_quick" if args.quick else "results")
    if args.step == "sheet":
        sheet(out)
    elif args.step == "note":
        fill_note(out)
    else:
        summarize(out, args.quick)
