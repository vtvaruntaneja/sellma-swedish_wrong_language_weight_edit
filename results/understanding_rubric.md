# Rubric: did the reply understand the prompt, and is its language label right?

The first version (below the line "Version 1") was written before any test reply was read. This version was written
on 2026-11-19, before any 512-token output existed. At that point the old --quick run had shown aggregate results
(not reply text) for every condition on the first 10 items of each test file. Those items are left out of the main
analysis. The rubric applies to `understanding_sheet.jsonl`, which `analyze.py sheet` builds.

**Items.** Every reply to a Swedish prompt that is not Swedish:
- from the base model's dev run at 128 tokens (whole-reply label, as run);
- from the base model's dev run at 512 tokens (two-halves rule: Swedish only if both halves are classified Swedish);
- from every test condition (two-halves rule).

Also every test reply that is Swedish on a prompt where the base model's reply is not (a "rescued" reply).

Items are deduplicated by (prompt id, reply text), shuffled with seed 0, and shown without the condition that
produced them. The annotator sees the prompt, lingua's labels (first half / second half), and, for long replies,
the first 50 words of each half. A reply that was not split is shown up to its first 100 words.

**Label 1, `understood`:**
- **yes**: the reply addresses the prompt's actual task and topic, for example tips, a recipe, an explanation or a
  text of the requested kind about the requested subject. Language, factual accuracy, grammar and truncation are
  ignored.
- **no**: the reply is about a different topic or task, misreads a key word of the prompt (for example
  "tentaperiod", exam period, read as "umbrella period"), answers a different question, or says it cannot help
  because it did not grasp the request.
- **unclear**: the reply is too short, too degenerate (loops, gibberish) or too mixed to judge.

**Label 2, `label_ok`** (measurement check, added at the user's request):
- **yes**: lingua is right about Swedish vs not Swedish. For a non-Swedish item, the half flagged as not Swedish
  really is mostly in another language. For a rescued item, the reply really is Swedish throughout.
- **no**: lingua is wrong about Swedish vs not Swedish. For example, the flagged half is Swedish with a few odd
  words, like sv015, sv177 and sv083 at 128 tokens.
- **unclear**: too mixed, too short or too degenerate to say.

If the language is not Swedish but lingua names the wrong one (for example Danish labelled Norwegian), the label
stays `yes` and the `note` says so. "Real language switches" are the non-Swedish items with `label_ok = yes`.

**Annotator.** Claude (Opus 5.5), the AI research engineer that also built the experiment. Single pass, blind to
condition, but not blind to the reply's language or to the hypothesis. There is no second annotator, so these
labels are a documented judgment, not a measurement.

---

Version 1 (written before any test reply was read; kept for the record):

Items: every reply to a Swedish prompt that lingua did not classify as Swedish, from the base model's dev run and
from every test condition, plus every test reply that became Swedish on a prompt where the base model's reply was
not Swedish. Deduplicated by (prompt id, reply text), shuffled with seed 0, shown without the condition. Labels yes /
no / unclear for "understood", with the definitions above, except that truncation at 128 tokens was named as ignored.
