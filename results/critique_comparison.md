# Comparison: my critique (results/critique_own.md) vs the researcher's notes (my_notes.md)

`critique_own.md` was written before `my_notes.md` was opened and is left unchanged. Updates to my view are recorded
here, and only where new evidence supports them. Every claim in both documents was checked against the blog
(https://rikiyat.github.io/blog/weight-space-programming/), its data files, the released logs
(https://github.com/RikiyaT/lm_understand-logs) and the Qwen3.5-0.8B model card
(https://huggingface.co/Qwen/Qwen3.5-0.8B, README at revision 2fc06364715b967f1860aea9cf38778875588b17).

## Where we agree (same evidence, verified)

| Point | Verified against |
|---|---|
| Exp. 3: copying heads found exactly from the middle blocks, without running the model | blog Exp. 3 table and Figure 1 |
| Exp. 7: weights alone failed in all 4 attempts; success needed the attention tool, which runs the models | blog Exp. 7 text |
| Exp. 8 is feedback-guided: up to 50 forward passes on GSM8K/MATH training problems; strength tried at 1, 2, 3, 4, final 6, each step kept on measured gains | edit log entries 04 to 17; `trajectories.json` |
| With 10 or 25 passes, looping barely changed: 87.2% and 86.4% of MATH-500 answers hit the token limit (base 93.0%) | blog Exp. 8 table |
| Exp. 8's gain is unblocking: 85/96 practice answers hit the limit; 576 of 579 rescues were previously cut off; Gemma 3 270M showed no meaningful gain | blog Exp. 8 text and "Additional experiment"; agent's final report |
| Small samples (4 games, 3 shapes, 3 character LMs, 3 Gemma pairs) | blog setups |
| Hints in prompts: Exp. 1 names the input channels; Exp. 3 lists the three answer options | blog Exp. 1 setup; Exp. 3 prompt ("Python code, romaji or English?") |
| Turning thinking off "raises the base score", but no number is given | blog, Response to Comments |

## Where we differ, and who has the better evidence

1. **Exp. 4 as "strongest evidence".** Your notes list it as strong, with the caveat that the agent also had the
   scene and one failing episode. The caveat is right. My view is that the result is real (the 0 → 100% change was
   measured), but it shows the agent can *write a controller into spare units*, not that it *repaired the trained
   policy*. The 57-line edit zeroes and rewrites units 122 to 127 of the first actor layer, and fewer units later.
   The blog's own description is "a small separate path ... once the green wheel is past the red strip, the path
   switches off and the trained network drives the rest". The task, though, asked the agent to "fix only its
   cause". **Better evidence: mine, on what kind of change it is; yours, that a large verified change happened.**
   One small correction to your notes: the base policy was measured on 256 episodes ("never succeeds in 256
   episodes"). Only the edited policy was measured on 1,024.

2. **Exp. 8's random-direction control.** My critique called it weak: one draw, run once mid-search. I checked the
   released logs (`exp8_math/runs/p50c/work_notes/experiment_history.md`, line 69): one norm-matched random direction,
   orthogonal to the repetition direction, on 64 problems × 2 samples. The result was 14/128 → 16/128, +1.56 pts,
   SE 3.52, with 8 rescues and 6 losses. It was matched to the intermediate full-direction strength-4 edit, not the
   final 64-neuron strength-6 edit. **Your point is better supported than my critique allowed.** The targeted edit's
   gains at that stage were tens of points (strength 2: 8.2% → 43.0% on new problems, edit-log entry 09), so a
   single 128-sample control is enough to rule out "any edit of this size would do it". **I update my view.** My
   remaining caveat is narrower: the control does not cover the final edit, and one draw says nothing about the
   spread across random directions.

3. **Which Exp. 8 evidence is strongest.** Your notes cite the random control and the layer-isolation tests. I would
   put first the frozen held-out confirmation: 1,152 practice problems not used to choose the edit, +49.8 pts
   (SE 1.5), 579 rescues against 5 regressions. Next comes MATH-500, which the agent never saw. Both test
   generalisation, not just specificity. We do not conflict here; this is a difference of emphasis.

4. **Model-card temperature.** You say the card recommends temperature 1.0 for thinking mode on text, while the blog
   used 0.6. **Correct** (card "Best Practices", README lines 707 and 935-936: temperature=1.0, top_p=0.95, top_k=20,
   presence_penalty=1.5). One nuance: the card's benchmark notes (README line 577) say Qwen's own thinking-mode
   evaluations for the non-video vision-language benchmarks used "the same sampling configuration but with
   temperature=0.6". That is exactly the blog's protocol (0.6 / 0.95 / 20 / 1.5). For the language benchmarks Qwen
   used temperature 1.0 (line 285). So the blog's setting is one Qwen itself used, but not the one it recommends
   for text. Whether 0.6 rather than 1.0 changes how often the model loops is untested by the blog and by us
   (unverified). The card's warning (line 856) says loops happen even "when using the recommended sampling
   parameters".

5. **Your other model-card points are verified as stated.** The 0.8B model is "more prone to entering thinking
   loops compared to other Qwen3.5 models" (line 856, thinking mode, recommended settings). It "operates in
   non-thinking mode by default" (line 585), and the chat template defaults to an empty think block. One number
   partly fills the "no number" gap: Qwen's card reports **IFEval 52.1 for Qwen3.5-0.8B in non-thinking mode**,
   against the blog's thinking-mode base 21.0 and edited 62.3. The protocols differ, so this is indicative only.

## Points only in your notes, accepted

- **Only one editing agent was tested** (GPT-6 Astra in every experiment). Correct. Agent-to-agent variance is
  unknown, on top of run-to-run variance.
- **It is a blog post, not a peer-reviewed paper.** Correct. The page shows "Published: Not published yet" and
  "DOI: No DOI yet".

## Points only in my critique (evidence in critique_own.md)

- The Exp. 8 task prompt names repetition as the example failure to fix and prescribes the method: diagnose,
  localize causally, dose-response, paired held-out tests (`exp8_math.json`).
- Exp. 8 forward passes return residual-stream and MLP activations, so localization was activation-derived.
- The three budgets found three different edits: 10 passes, 30 rank-one edits in layers 8 to 22; 25 passes, one
  attention head zeroed; 50 passes, 64 MLP columns in layers 15 and 19. So the "issue found" is not stable, and
  there is one run per budget.
- "0.016% of weights" is a poor size measure. Strength 6 reverses and amplifies the component (1 − 6 = −5), and
  the denominator includes the 254M-entry tied embedding.
- Regressions were measured only where loop removal helps. Base GPQA Diamond is 22.7%, below the 25% chance rate
  of a 4-option test.
- Exp. 5 and 6 compile hand-designed programs: game source code was given, and a parameterised Go heuristic was
  tuned by keeping changes that won more games.
- The "not a method" framing sits awkwardly with the comparative TL;DR. The listed replications are unverified.

## Effect on the experimental plan

**None.** The comparison supports choices already in the plan:
- name our method a feedback-guided, activation-derived weight edit and count the feedback budget;
- use 3 norm-matched random directions, not one, evaluated on the same held-out test prompts as the real edit;
- include prompt and sampling baselines, the empty-system-prompt control and a LoRA baseline;
- measure regressions where the target behaviour plays no role (English replies, exact-answer accuracy, Wikipedia
  perplexity).

The plan proceeds to step 5 unchanged.
