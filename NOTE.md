# Can a small weight edit stop wrong-language replies to Swedish prompts?

No. My edit did no better than random edits of the same size. A Swedish system prompt and a small fine-tune each
raised the Swedish-reply rate by about 11 points, but each damaged something else. Most wrong-language replies also
misread the prompt, so the main problem looks like a missing skill, not a blocker.

I tested the kind of edit the blog's Experiment 8 used (a feedback-guided, activation-derived weight edit), not
whether an AI can find edits by reading weights. The appendix gives the source file of every number.

## 1. Critique of the blog

The blog shows that an AI agent can find useful edits when it can run the model and measure the results. It does
not show that an AI can read a real language model's behaviour from its weights alone.

- Experiment 3 is the best evidence for weight reading. From the weights of tiny character models, the agent named
  the "copying heads" (parts that repeat earlier text), and a separate measurement confirmed them.
- Experiment 8 shows unblocking. The edit gained 49.8 points (SE 1.5) on 1,152 held-out problems, and 576 of its 579
  rescues had been cut off by thinking loops. The same procedure did not help Gemma 3 270M, which rarely loops.
- Experiment 8 is not weight reading. The loops were found in generated text, located with activations, and the
  edit was tuned on measured accuracy. The only weights-only test on a language model (Experiment 7) failed.
- Simple baselines are missing, and the random control was a single draw. "0.016% of weights" counts touched
  numbers, not the size of the change.

My full critique is in `results/critique_own.md`, and the comparison with my notes is in
`results/critique_comparison.md`.

## 2. Hypothesis

My bet was that wrong-language replies are a blocker, like Qwen's thinking loops. The model can answer in Swedish,
but something small pushes it into English, Norwegian or Danish.

- The bet is supported if the edit beats the base model and random edits of the same size on held-out prompts,
  regressions are small, and no prompt or sampling change matches it.
- The bet is challenged if random edits do as well, a prompt or sampling change does as well or better, other
  abilities get worse, or the failure barely exists.

## 3. Setup

I compared one selected weight edit with random edits and simple alternatives, on Swedish test prompts that I did
not use to choose the edit.

- **Model.** Qwen2.5-0.5B-Instruct (revision `7ae5576`). The planned Qwen3.5-0.8B gave 0 wrong-language replies to
  40 pilot prompts in the first 128 tokens (Wilson 95% CI 0-8.8%), so I switched. This says nothing about longer
  replies.
- **Data and language check.** Claude wrote 300 Swedish prompts (150 dev, 150 test), 150 English prompts and 60
  questions with exact answers. A reply counts as Swedish only if lingua, a language detector that was right on 100%
  of 200 test paragraphs, classifies both of its halves as Swedish. Replies are greedy, up to 512 new tokens.
- **The edit.** It scales what four layers write into the model's running internal state along one direction
  (W ← W − α r rᵀ W). Its free parameters are one 896-number direction and one strength. It changes 4.18% of all
  weights. That cannot be compared with the blog's 0.016%, because my edit touches every entry of 8 matrices, but
  only along one direction.
- **Dev work.** A fixed rule picked one of 16 dev settings. The feedback budget (dev runs of edited models) was 17.
- **Conditions.** Base; the edit; 5 random directions of the same size; the system prompt "Svara alltid på
  svenska."; an empty system prompt (the template otherwise adds an English one); the repo's sampling settings (3
  seeds); and LoRA, a small add-on to the weights trained on the model's own Swedish replies to dev prompts.
- **Test-set use.** An early smoke test read the first 10 items of each test file, so my main results leave them
  out: 140 Swedish prompts, 90 English prompts, 50 questions and 90 Wikipedia paragraphs. The full set agrees.

## 4. Results

The edit did not help. The simple fixes did better, but each had a cost.

This table shows each condition on the main test set. "Fixed / newly broken" counts Swedish prompts that changed
outcome compared with the base model. QA accuracy is on the questions with exact answers.

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

- **The edit.** It scored −2.9 points against base [−10.0, +3.6], p = 0.54. So a gain above 3.6 points is ruled out
  at 95% confidence. Against the random edits it scored −2.7.
- **Churn.** The edit fixed 10 prompts and broke 14. Each random edit fixed 7 to 12 and broke 9 to 11. So the edit
  behaves like a random disturbance of the same size. The table below shows this per random draw.
- **Dev.** I tried 16 settings in 2 rounds. The best setting in each round beat base by one prompt.
- **Swedish system prompt.** +11.4 points (p = 0.007). It broke 6 English prompts. Four of them now get Swedish
  replies, which is the model following the instruction. The other 2 are real damage: en016 switches language
  partway, and en096 is a "Catnip" loop.
- **LoRA.** +11.4 points (p = 0.011). But perplexity on Swedish Wikipedia went from 47.2 to 123.5. Perplexity is how
  surprised the model is by real text; lower is better. The edit only moved it to 49.2. A possible cause is 3
  training epochs on the model's own replies (final training loss 0.0415). Its QA gain (+10, p = 0.0625) is not
  significant.
- **Others.** The empty system prompt gave +6.4 (p = 0.14) and sampling gave +5.0. Neither is significant.

This table compares the edit's churn with each random edit.

<!-- table:test main: churn, edit vs random edits -->
| Weights | fixed | newly broken | changed (fixed + broken) | net |
|---|---|---|---|---|
| edited | 10 | 14 | 24 | -4 |
| random_s1 | 9 | 11 | 20 | -2 |
| random_s2 | 7 | 9 | 16 | -2 |
| random_s3 | 11 | 10 | 21 | +1 |
| random_s4 | 12 | 10 | 22 | +2 |
| random_s5 | 10 | 10 | 20 | +0 |
<!-- /table:test main: churn, edit vs random edits -->

## 5. Interpretation: unblocking or new skill?

The failures are mostly a missing skill. Only a few look like a blocker that hides an ability the model has.

I read every wrong-language reply without knowing which condition produced it. For each one I noted whether lingua's
label was right and whether the reply understood the prompt. This table shows the base model's 32 test failures.

<!-- table:test main: base failures by label and understanding -->
| Base failure is a | understood the prompt | misread it | unclear | total |
|---|---|---|---|---|
| real wrong-language reply | 5 | 17 | 1 | 23 |
| detector error (the reply is Swedish) | 2 | 3 | 3 | 8 |
| unclear label | 0 | 0 | 1 | 1 |
<!-- /table:test main: base failures by label and understanding -->

- Of the 23 real wrong-language replies, 17 also misread the prompt. Only 5 understood it.
- 8 "failures" were detector errors: broken or looping Swedish that lingua called another language.
- This matches the blog's Gemma result. An edit can remove a blocker, but it cannot add a missing skill.
- The clearest single cause I found is the English default system prompt. Of 16 English failures, the Swedish system
  prompt fixed 13, the empty one fixed 10, and LoRA fixed all 16.

## 6. Limitations

- The understanding and label marks were made by Claude alone, with no human check.
- I made many comparisons without correcting for multiple tests. My main conclusion rests on two of them: edit vs
  base and edit vs random.
- Base scored 84.7% Swedish on dev and 77.1% on test. The 95% intervals overlap (78.0-89.6% and 69.5-83.3%), so the
  gap may be chance.
- I left out the 10 test items seen early. The full set agrees.
- I tested the blog's kind of edit, not whether an AI can find edits by reading weights.
- I skipped LoRA's own dev check, because the process stalled with memory full after training.
- Time: the assignment window was about 1.25 days (18 Nov 19:20 to the end of 19 Nov CET). With more time I would be
  more confident in these conclusions. I would add a larger test set (this one can only detect differences of about
  9.5 points or more), a human Swedish speaker to check the labels, a second model, the skipped LoRA dev check, and a
  multiple-comparison correction.

## 7. Recommendation and next experiment

SE.LLMA should not prioritise this kind of weight editing for Swedish language-choice failures. Fix template
defaults first. Improve Swedish skill with data and fine-tuning, while checking general Swedish quality. Use targeted
edits only where one cause is confirmed.

The next experiment I would run is a targeted edit against the English-default-prompt cause, compared with simply
removing that prompt. I would add labels from a human Swedish speaker.

## Appendix: where each number comes from

This table lists each number in the text above, the file it comes from, and the field. The values are read from the
files by `analyze.py note`.

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

The blog's numbers in section 1 come from the blog and its data files and logs, as cited in
`results/critique_own.md` and `results/critique_comparison.md`. The time window in section 6 is the researcher's own
statement, not a results file. The data sizes in section 3 are the line counts of the files in `data/`.
