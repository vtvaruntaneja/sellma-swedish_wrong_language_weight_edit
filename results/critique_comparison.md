# Comparison: Claude's critique (results/critique_own.md) and my notes (my_notes.md)

*Written by Claude after it had finished its own critique. On 20 Nov 2026 the wording of this file was simplified
for readability; the content is unchanged. The original text is in git commit `e1b512c`.*

`critique_own.md` was written before `my_notes.md` was opened. Its content has not changed since (only its wording
was simplified on 20 Nov 2026). Any change of view is recorded here, and only where new evidence supports it. Every
claim in both documents was checked against the blog (https://rikiyat.github.io/blog/weight-space-programming/), its
data files, the released logs (https://github.com/RikiyaT/lm_understand-logs) and the Qwen3.5-0.8B model card
(https://huggingface.co/Qwen/Qwen3.5-0.8B, README at revision 2fc06364715b967f1860aea9cf38778875588b17).

## Where we agree

This table lists the points where both documents say the same thing, and where each one was checked.

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

1. **Is Experiment 4 among the strongest evidence?** Your notes list it as strong, with the caveat that the agent
   also had the scene and one failing episode. That caveat is right. My view is that the result is real (the change
   from 0 to 100% was measured), but it shows the agent can *write a controller into spare units*, not that it
   *repaired the trained policy*. The 57-line edit zeroes and rewrites units 122 to 127 of the first actor layer, and
   fewer units later. The blog itself describes it as "a small separate path ... once the green wheel is past the red
   strip, the path switches off and the trained network drives the rest". Yet the task asked the agent to "fix only
   its cause". **Better evidence: mine, on what kind of change it is; yours, that a large, verified change
   happened.** One small correction to your notes: the base policy was measured on 256 episodes ("never succeeds in
   256 episodes"). Only the edited policy was measured on 1,024.

2. **Experiment 8's random-direction control.** My critique called it weak: one draw, run once in the middle of the
   search. I then checked the released logs (`exp8_math/runs/p50c/work_notes/experiment_history.md`, line 69). There
   is one random direction of the same size, at right angles to the repetition direction, tested on 64 problems × 2
   samples. The result was 14/128 → 16/128: +1.56 points, SE 3.52, with 8 rescues and 6 losses. It was matched to the
   in-between edit (the full direction at strength 4), not to the final edit (64 neurons at strength 6). **Your point
   is better supported than my critique allowed.** At that stage the targeted edit gained tens of points (strength 2:
   8.2% → 43.0% on new problems, edit-log entry 09). So even one control with 128 samples is enough to rule out "any
   edit of this size would do it". **I update my view.** My remaining caveat is narrower: the control does not cover
   the final edit, and one draw says nothing about how much random directions vary.

3. **Which Experiment 8 evidence is strongest?** Your notes point to the random control and the layer-isolation
   tests. I would put the frozen held-out check first: 1,152 practice problems not used to choose the edit, +49.8
   points (SE 1.5), with 579 rescues against 5 regressions. MATH-500, which the agent never saw, comes next. Both test
   whether the gain carries over to new problems, not just whether the edit is specific. We don't disagree here; we
   only put the weight in different places.

4. **The model card's temperature.** You say the card recommends temperature 1.0 for thinking mode on text, while
   the blog used 0.6. **That is correct** (card "Best Practices", README lines 707 and 935-936: temperature=1.0,
   top_p=0.95, top_k=20, presence_penalty=1.5). One nuance: the card's benchmark notes (README line 577) say that
   Qwen's own thinking-mode tests on the non-video vision-language benchmarks used "the same sampling configuration
   but with temperature=0.6". That is exactly the blog's setup (0.6 / 0.95 / 20 / 1.5). For the language benchmarks,
   Qwen used temperature 1.0 (line 285). So the blog's setting is one Qwen itself used, but not the one it recommends
   for text. Whether 0.6 instead of 1.0 changes how often the model loops has not been tested, by the blog or by us
   (unverified). The card's warning (line 856) says loops happen even "when using the recommended sampling
   parameters".

5. **Your other model-card points check out as stated.** The 0.8B model is "more prone to entering thinking loops
   compared to other Qwen3.5 models" (line 856, thinking mode, recommended settings). It "operates in non-thinking
   mode by default" (line 585), and the chat template starts with an empty think block. One number partly fills the
   "no number given" gap: Qwen's card reports **IFEval 52.1 for Qwen3.5-0.8B with thinking off**, against the blog's
   thinking-mode base of 21.0 and edited score of 62.3. The setups differ, so this only gives a rough idea.

## Points only in your notes, which I accept

- **Only one editing agent was tested** (GPT-6 Astra, in every experiment). Correct. We don't know how much
  different agents would vary, on top of how much runs vary.
- **It is a blog post, not a peer-reviewed paper.** Correct. The page shows "Published: Not published yet" and
  "DOI: No DOI yet".

## Points only in my critique (evidence in critique_own.md)

- The Experiment 8 task prompt names repetition as the example failure to fix, and lays out the method: diagnose,
  locate the cause, try several strengths, and use paired held-out tests (`exp8_math.json`).
- Experiment 8's forward passes return residual-stream and MLP activations, so the edit was located using
  activations.
- The three budgets found three different edits: with 10 passes, 30 rank-one edits in layers 8 to 22; with 25
  passes, one attention head switched off; with 50 passes, 64 MLP columns in layers 15 and 19. So the "problem found"
  is not stable, and there is one run per budget.
- "0.016% of weights" is a poor measure of size. Strength 6 reverses and amplifies the component (1 − 6 = −5), and
  the total it is a share of includes the 254M-entry tied embedding.
- Damage was measured only where removing loops helps. The base model's GPQA Diamond score is 22.7%, below the 25%
  you would get by guessing on a 4-option test.
- Experiments 5 and 6 write hand-designed programs into weights: the agent was given the game's source code, and it
  tuned a Go heuristic with settings by keeping changes that won more games.
- The "not a method" framing does not sit well with a summary that leads with comparisons. The listed replications
  are unverified.

## Effect on the experimental plan

**None.** The comparison supports choices that were already in the plan:
- call our method a feedback-guided, activation-derived weight edit, and count the feedback budget;
- use 3 random directions of the same size, not one, tested on the same held-out prompts as the real edit;
- include prompt and sampling baselines, the empty-system-prompt control and a LoRA baseline;
- measure damage where the target behaviour plays no part (English replies, exact-answer accuracy, Wikipedia
  perplexity).

The plan went on to step 5 unchanged.
