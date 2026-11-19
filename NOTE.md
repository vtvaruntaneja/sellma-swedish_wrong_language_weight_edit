# Can a small weight edit stop wrong-language replies to Swedish prompts?

**No.** My edit did no better than random edits of the same size. A Swedish system prompt and a small fine-tune
(LoRA) each raised Swedish replies by about 11 points, but each caused damage elsewhere. Most wrong-language replies
also misread the question, so the problem looks like weak Swedish, not a switch that an edit could flip. I tested the
kind of edit the blog's Experiment 8 used (found from activations, chosen with feedback), not whether an AI can find
edits by reading weights. The appendix gives the source of every number.

## 1. What I think of the blog

The blog shows an AI agent finding useful edits when it can run the model and measure the results. It does not show
an AI reading a real language model's behaviour from its weights alone.

- **Best evidence for weight reading: Experiment 3.** Without running tiny character models, the agent named their
  "copying heads", and a separate measurement confirmed them.
- **Experiment 8 removed a blocker, through trial and feedback.** It gained 49.8 points (SE 1.5) on 1,152 practice
  problems not used for tuning, but 576 of its 579 rescued answers had been cut off by thinking loops. Its strength
  was tuned on measured accuracy, and it did not help Gemma 3 270M, which rarely loops.
- **Weak spots.** The one weights-only test on a real language model (Experiment 7) failed. There are no simple
  baselines and only one random control. "0.016% of weights" counts the numbers touched, not the size of the change.

Claude's full critique, written before it saw my notes (`my_notes.md`), is in `results/critique_own.md`, and
`results/critique_comparison.md` compares the two.

## 2. Hypothesis

Small models sometimes answer Swedish questions in English, Norwegian or Danish, which makes a Swedish assistant
useless to its users. My bet was that this is a blocker, like the blog's loops, and that a targeted edit could remove
it. The bet is **supported if** the edit beats the base model and random edits of the same size on unseen prompts,
with little damage elsewhere, and no prompt or sampling change does as well. It is **challenged if** random edits or
simple changes do as well, other abilities get worse, or the problem barely exists.

## 3. Setup

- **Model.** Qwen2.5-0.5B-Instruct (revision `7ae5576`). The blog's Qwen3.5-0.8B gave 0 wrong-language replies to 40
  pilot prompts in the first 128 tokens (95% interval 0-8.8%), so I switched. That says nothing about longer replies.
- **Data and measure.** Claude wrote 300 Swedish prompts (150 for development, 150 for testing), 150 English prompts
  and 60 questions with exact answers. A reply counts as Swedish only if lingua, a language detector that was right
  on 100% of 200 check paragraphs, calls both of its halves Swedish. Replies are greedy (except for sampling), up to
  512 new tokens.
- **The edit.** W ← W − α r rᵀ W in four layers, with one 896-number direction plus one strength. It changes 4.18%
  of all weights (every entry of 8 matrices, but only along one direction), so it can't be compared with the blog's
  0.016%. A fixed rule picked it from the 8 settings of the second of two development rounds (16 settings in all),
  which took 17 runs of edited models (the feedback budget).
- **Test conditions.** The base model; the edit; 5 random directions of the same size; the system prompt "Svara
  alltid på svenska." ("Always answer in Swedish."); an empty system prompt (the template otherwise adds an English
  one); the model's own sampling settings, with 3 seeds; and LoRA, trained on the model's Swedish replies to
  development prompts.
- **Test set.** A smoke test saw the first 10 items of each test file early, so the main results use the rest: 140
  Swedish prompts, 90 English prompts, 50 questions and 90 Wikipedia paragraphs. The full set agrees.

## 4. Results

The edit did not help. The simple fixes helped more, at a cost. In the table, "fixed / newly broken" counts Swedish
prompts whose outcome changed compared with the base model.

<!-- table:test main: overview -->
| Condition | Swedish replies to Swedish prompts [95% CI] | fixed / newly broken vs base | English replies to English prompts | QA accuracy |
|---|---|---|---|---|
| base | 77.1% [69.5%, 83.3%] |  | 98.9% | 64.0% |
| edited | 74.3% [66.5%, 80.8%] | 10 / 14 | 100.0% | 66.0% |
| random, mean of 5 | 77.0% | 9.8 / 10.0 | 98.7% | 65.6% |
| sysprompt_sv | 88.6% [82.2%, 92.8%] | 24 / 8 | 92.2% | 70.0% |
| sysprompt_empty | 83.6% [76.6%, 88.8%] | 19 / 10 | 98.9% | 72.0% |
| sampling, mean of 3 | 82.1% | 14.0 / 7.0 | 99.3% | 64.0% |
| lora | 88.6% [82.2%, 92.8%] | 26 / 10 | 98.9% | 74.0% |
<!-- /table:test main: overview -->

- **The edit** scored −2.9 points against the base model (95% interval −10.0 to +3.6, p = 0.54), so a gain above 3.6
  points is ruled out. Against the average random edit it scored −2.7. It fixed 10 prompts and broke 14, while each
  random edit fixed 7 to 12 and broke 9 to 11, so it acts like a random nudge (full table in `results/summary.md`).
  In development, the best setting of each round was one prompt better than the base model (133 vs 132 of 150 at
  128 tokens, and 128 vs 127 at 512 tokens).
- **The Swedish system prompt** gave +11.4 (p = 0.007), but it broke 6 English prompts. 4 of them now get Swedish
  replies, which is the model obeying the instruction. The other 2 are real damage: en016 switches partway, and
  en096 repeats "Catnip".
- **LoRA** gave +11.4 (p = 0.011). But its perplexity on Swedish Wikipedia (how surprised it is by real text; lower
  is better) rose from 47.2 to 123.5, while the edit moved it only to 49.2. A possible cause is 3 epochs on the
  model's own replies, down to a training loss of 0.0415. Its QA gain (+10, p = 0.0625) is not significant.
- **Not significant:** the empty system prompt (+6.4, p = 0.14) and sampling (+5.0).

## 5. Blocker or missing skill?

Mostly a missing skill. Claude labelled every wrong-language reply without knowing its condition: was lingua's label
right, and did the reply understand the question? This table covers the base model's 32 test failures.

<!-- table:test main: base failures by label and understanding -->
| Base failure is a | understood the prompt | misread it | unclear | total |
|---|---|---|---|---|
| real wrong-language reply | 5 | 17 | 1 | 23 |
| detector error (the reply is Swedish) | 2 | 3 | 3 | 8 |
| unclear label | 0 | 0 | 1 | 1 |
<!-- /table:test main: base failures by label and understanding -->

Of the 23 replies that really were in the wrong language, 17 also misread the question, and only 5 understood it.
Another 8 were detector errors on broken or looping Swedish. This fits the blog's Gemma result: an edit can remove a
blocker, but it cannot add a missing skill. The clearest single cause is the English default system prompt. Of the
16 English failures, the Swedish system prompt fixed 13, the empty one 10, and LoRA all 16.

## 6. Limitations

- The labels come from Claude alone, with no human check.
- I made many comparisons without a correction for multiple tests. The conclusion rests on two: the edit against the
  base model, and the edit against random edits.
- The base model scored 84.7% on development prompts and 77.1% on test prompts, but the 95% intervals overlap
  (78.0-89.6% and 69.5-83.3%), so the gap may be chance.
- I tested the blog's kind of edit, not weight reading. I skipped LoRA's development check, because its process
  stalled with memory full.
- The assignment window was under a day and a half. With more time I would be more confident. I would add a larger
  test set (this one detects only differences of about 9.5 points or more), a human Swedish speaker to check the
  labels, a second model, the LoRA check, and a correction for multiple tests.

## 7. Recommendation and next experiment

SE.LLMA should not prioritise this kind of weight editing for Swedish language-choice problems. Fix template defaults
first, such as the hidden English system prompt. Improve Swedish skill with data and fine-tuning, while checking that
general Swedish quality holds. Keep targeted edits for cases with one confirmed cause. Next, I would test a targeted
edit against the English default prompt, compared with simply removing it, and have a Swedish speaker check the
labels.

## Appendix: where each number comes from

This is a reference table, not part of the two-page note. `analyze.py note` reads each value from its file.

<!-- table:appendix -->
| Number in the text | File | Field | Value in the file |
|---|---|---|---|
| 0 of 40 wrong-language replies, Qwen3.5 pilot | `results/summary.json` | `pilot.qwen35.n_wrong` | 0 |
| Wilson 95% CI 0-8.8% | `results/summary.json` | `pilot.qwen35.wrong_rate_wilson95` | [0, 0.08762] |
| first 128 tokens (pilot) | `results/summary.json` | `pilot.qwen35.max_new_tokens` | 128 |
| revision 7ae5576 | `results/test_config.json` | `revision` | 7ae557604adf67be50417f59c2c2f167def9a775 |
| lingua right on 100% | `results/langid_check_summary.json` | `all_languages/text.accuracy` | 1 |
| of 200 paragraphs | `results/langid_check_summary.json` | `all_languages/text.n` | 200 |
| up to 512 new tokens | `results/test_config.json` | `max_new_tokens.sv` | 512 |
| one 896-number direction and one strength | `results/test_config.json` | `edit_sizes.edited.free_parameters` | 897 |
| 4.18% of all weights | `results/test_config.json` | `edit_sizes.edited.percent_of_weights_changed` | 4.179 |
| 8 matrices | `results/test_config.json` | `edit_sizes.edited.matrices` | 8 |
| feedback budget 17 | `results/dev512_selection.json` | `feedback_budget.total_dev_evaluations_of_edited_models` | 17 |
| 5 random directions | `results/test_config.json` | `random_direction_seeds` | [1, 2, 3, 4, 5] |
| 3 sampling seeds | `results/test_config.json` | `sampling_seeds` | [1, 2, 3] |
| 140 Swedish prompts | `results/summary.json` | `subsets.main.conditions.base.n_sv` | 140 |
| 90 English prompts | `results/summary.json` | `subsets.main.conditions.base.n_en` | 90 |
| 50 questions | `results/summary.json` | `subsets.main.conditions.base.n_qa` | 50 |
| 90 paragraphs (Swedish part) | `results/summary.json` | `subsets.main.wikipedia_ppl.base.sv.n` | 45 |
| 90 paragraphs (English part) | `results/summary.json` | `subsets.main.wikipedia_ppl.base.en.n` | 45 |
| edit vs base: -2.9 points | `results/summary.json` | `subsets.main.paired_vs_base.edited.sv.diff` | -0.02857 |
| edit vs base: [-10.0, +3.6] | `results/summary.json` | `subsets.main.paired_vs_base.edited.sv.ci95` | [-0.1, 0.03571] |
| edit vs base: p = 0.54 | `results/summary.json` | `subsets.main.paired_vs_base.edited.sv.mcnemar_p` | 0.5413 |
| edit vs random edits: -2.7 | `results/summary.json` | `subsets.main.paired_vs_edited.random.sv.diff` | -0.02714 |
| edit fixed 10 | `results/summary.json` | `subsets.main.paired_vs_base.edited.sv.gained` | 10 |
| edit broke 14 | `results/summary.json` | `subsets.main.paired_vs_base.edited.sv.lost` | 14 |
| dev round 1: base (132 of 150 Swedish) | `results/dev_selection.json` | `base.swedish_rate` | 0.88 |
| dev round 1: best setting L12-15_a1, one prompt more | `results/dev_selection.json` | `combos.4.swedish_rate` | 0.8867 |
| dev round 2: base | `results/dev512_selection.json` | `base.swedish_rate` | 0.8467 |
| dev round 2: selected setting fail_L12-15_a2, one prompt more | `results/dev512_selection.json` | `settings.3.swedish_rate` | 0.8533 |
| Swedish system prompt: +11.4 | `results/summary.json` | `subsets.main.paired_vs_base.sysprompt_sv.sv.diff` | 0.1143 |
| Swedish system prompt: p = 0.007 | `results/summary.json` | `subsets.main.paired_vs_base.sysprompt_sv.sv.mcnemar_p` | 0.007 |
| it broke 6 English prompts | `results/summary.json` | `subsets.main.paired_vs_base.sysprompt_sv.en.lost` | 6 |
| 4 Swedish replies, en016, en096 | `results/summary.json` | `subsets.main.english_newly_broken.sysprompt_sv` | [en016 en/other, en035 sv/sv, en069 sv/sv, en072 sv/sv, en079 sv/sv, en096 other/other] |
| LoRA: +11.4 | `results/summary.json` | `subsets.main.paired_vs_base.lora.sv.diff` | 0.1143 |
| LoRA: p = 0.011 | `results/summary.json` | `subsets.main.paired_vs_base.lora.sv.mcnemar_p` | 0.01133 |
| Swedish Wikipedia perplexity, base: 47.2 | `results/summary.json` | `subsets.main.wikipedia_ppl.base.sv.ppl` | 47.2 |
| Swedish Wikipedia perplexity, LoRA: 123.5 | `results/summary.json` | `subsets.main.wikipedia_ppl.lora.sv.ppl` | 123.5 |
| Swedish Wikipedia perplexity, edit: 49.2 | `results/summary.json` | `subsets.main.wikipedia_ppl.edited.sv.ppl` | 49.19 |
| 3 training epochs | `results/lora_summary.json` | `epochs_run` | 3 |
| final training loss 0.0415 | `results/lora_summary.json` | `loss_log.53.loss` | 0.04153 |
| LoRA QA gain: +10 | `results/summary.json` | `subsets.main.paired_vs_base.lora.qa.diff` | 0.1 |
| LoRA QA gain: p = 0.0625 | `results/summary.json` | `subsets.main.paired_vs_base.lora.qa.mcnemar_p` | 0.0625 |
| empty system prompt: +6.4 | `results/summary.json` | `subsets.main.paired_vs_base.sysprompt_empty.sv.diff` | 0.06429 |
| empty system prompt: p = 0.14 | `results/summary.json` | `subsets.main.paired_vs_base.sysprompt_empty.sv.mcnemar_p` | 0.136 |
| sampling: +5.0 | `results/summary.json` | `subsets.main.paired_vs_base.sampling.sv.diff` | 0.05 |
| 32 base test failures | `results/summary.json` | `subsets.main.understanding.test_base.wrong_language_replies` | 32 |
| 23 real, 17 misread, 5 understood, 8 detector errors | `results/summary.json` | `subsets.main.understanding.test_base.label_ok_by_understood` | yes/yes: 5, yes/no: 17, yes/unclear: 1, no/yes: 2, no/no: 3, no/unclear: 3, unclear/yes: 0, unclear/no: 0, unclear/unclear: 1 |
| English failures fixed by the Swedish system prompt: 13 of 16 | `results/summary.json` | `subsets.main.base_failures_fixed_by_language.sysprompt_sv.en` | base_failures: 16, fixed: 13 |
| English failures fixed by the empty system prompt: 10 | `results/summary.json` | `subsets.main.base_failures_fixed_by_language.sysprompt_empty.en` | base_failures: 16, fixed: 10 |
| English failures fixed by LoRA: 16 | `results/summary.json` | `subsets.main.base_failures_fixed_by_language.lora.en` | base_failures: 16, fixed: 16 |
| base on dev: 84.7% | `results/dev512_selection.json` | `base.swedish_rate` | 0.8467 |
| base on dev: 78.0-89.6% | `results/summary.json` | `dev512_base_swedish_wilson95` | [0.7804, 0.8956] |
| base on test: 77.1% | `results/summary.json` | `subsets.main.conditions.base.swedish_rate` | 0.7714 |
| base on test: 69.5-83.3% | `results/summary.json` | `subsets.main.conditions.base.swedish_rate_wilson95` | [0.6952, 0.8332] |
| detectable difference about 9.5 points | `results/summary.json` | `subsets.main.power.mde_80pct_power_at_observed_discordance` | 0.0954 |
<!-- /table:appendix -->
