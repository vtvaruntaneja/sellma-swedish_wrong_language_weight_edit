# Critique of "Sparks of Weight-Space Programming"

*Written by Claude before it read my notes (`my_notes.md`). On 20 Nov 2026 the wording was simplified for
readability; the content is unchanged. The original text is in git commit `e1b512c`.*

**Source.** Blog post by Takehi, Teoh, Zhang and Khattab, dated 2 Nov 2026, with a "Response to Comments" section
added on 12 Nov 2026 (https://rikiyat.github.io/blog/weight-space-programming/). The details on Experiment 8 come from
the blog's own data files: the edit logs
(https://rikiyat.github.io/blog/weight-space-programming/assets/mathedit/trajectories.json), the agent's prompt and
final answers (https://rikiyat.github.io/blog/weight-space-programming/assets/traces/exp8_math.json) and the
benchmark results (https://rikiyat.github.io/blog/weight-space-programming/assets/mathedit/benchmarks.json). Every
number below is one the blog reports. None of them are ours.

## 1. The key question: what could the editing agent see?

The assignment asks us to separate two things: reasoning about the weights, and improving an edit by running the
model again and again. The blog's experiments sit at very different points between those two. This table shows what
the agent had in each experiment, and whether it got feedback from running the model.

| Exp. | Task | What the agent had | Model feedback? |
|---|---|---|---|
| 1 | Which game a policy plays | all weights as text, channel meanings described | none |
| 2 | Which shape a diffusion MLP draws | all weights as text | none |
| 3 | Training language and copying heads of a 4-layer char LM | middle blocks + Python (up to 100 runs), forward passes forbidden | none |
| 4 | Fix Kinetix car policy | weights, scene description, **one recorded episode** | one trajectory |
| 5 | Write MinAtar policies from scratch | **game rules and source code**, zero weights | none |
| 6 | Write a Go network | Python sandbox, **0 to 50 forward passes** of up to 128 games each | yes; "keeps a change only if it wins more" |
| 7 | Which LM is sycophantic | weights + Python; **failed in all 4 weights-only attempts**; succeeded only with attention patterns from up to 20 runs | yes (activations) |
| 8 | Make Qwen3.5-0.8B better at math | Python kernel + **10/25/50 forward passes**, each up to 512 generations of up to 32,768 tokens or 1,024 teacher-forced texts, **with residual-stream and MLP activations returned** | yes, heavy |

Only Experiments 1 to 3 are reasoning without running the model. (Experiment 5 is writing a program from a
description, with no weights at all.) The two results that matter most for real language models, Experiments 7 and
8, both depend on running the model.

## 2. The strongest evidence

1. **Experiment 3: copying heads.** The agent looked at four transformer blocks of a 7.2M-parameter character model.
   It had Python but could not run the model ("We checked every run"). It named exactly the induction heads (the
   "copying" heads, which repeat earlier text) that the authors found separately by running the model. It did this in
   all three models and named no other head. Because the answer was checked by a separate measurement, this is a real
   test. The caveats: the method is a known one (weight composition, Elhage et al. 2021, as the blog notes); the
   models are tiny; and the training-language question was a choice between three options given in the prompt,
   answered with confidence of only 0.48 to 0.62.
2. **Experiment 8: a held-out check.** The 50-pass edit was frozen and then tested on 1,152 practice problems that
   were not used to choose it. It gained +49.8 points (SE 1.5), with 579 answers going from wrong to right and 5 from
   right to wrong (the agent's final report in `trajectories.json`). On MATH-500, which the agent never saw, the
   score went from 7.9% to 48.1%. Under that test setup, the edit's effect on looping is real and large.
3. **The gain is mostly "unblocking", and the authors say so.** 576 of the 579 rescued answers had been cut off at
   the token limit before, and accuracy on answers that already finished did not change (agent's final report). In
   the "Additional experiment", the same procedure could not improve Gemma 3 270M, a model that rarely loops
   (MATH-500: 6.8% before, 7.0 to 7.7% after). Together, these are the post's best evidence for the difference
   between removing a blocker and adding a skill. But it is one non-looping model, of a different size and family, so
   it suggests rather than proves.
4. **Openness.** The logs, the agent's prompt, the failures (the weights-only part of Experiment 7, and Gemma) and the
   edited model are all public (https://github.com/RikiyaT/lm_understand-logs,
   https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming). That is what makes the points below
   checkable.

## 3. The most important limitations

1. **The summary at the top calls Experiment 8 weight reading, but the log shows a search guided by feedback.** The
   TL;DR says the agent "identifies an issue in its weights ... and fixes this by natively editing 0.016% of the
   weights". In the log, the problem was found in the generated text (turn 3: "85/96 outputs were cut off"). Its
   location was then found with activations from running the model ("Matched productive/repetitive contexts
   localized the relevant MLP writers"). The final edit removes a direction in the style of Arditi et al. (2024), a
   paper the blog itself cites. So it was derived from activations and tuned with feedback, not read off the weights.
2. **The prompt supplied the hypothesis and the method.** The Experiment 8 task prompt (`exp8_math.json`) says: "for
   example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives
   that repetition counts". Its "HOW TO WORK" section also lays out the steps: diagnose first, locate the cause, try
   several strengths to see the dose-response, and use paired tests on held-out data. The agent carried out a search
   that the experimenters had outlined. It did not discover the failure from the weights.
3. **The strength and the parts edited were tuned with feedback from the model.** In the 50-pass log, the strength
   went from 1 to 2, 3 and 4, then to a 64-neuron subset at strength 6 (edit-log entries 04 to 17), with accuracy
   measured after each step. Roughly the first 33 passes went on diagnosis and tuning, and the rest on the frozen
   check (the turn-by-turn `used` counter in `trajectories.json`). This is hill-climbing, with an LLM choosing each
   step.
4. **The result depends on the budget and on the run, and the "problem found" changes between runs.** The three
   budgets produced three different edits:
   - 10 passes: 30 rank-one edits (each changes a matrix along one direction) in layers 8 to 22, +3.7 points on
     practice problems, called "modest" by the agent, with a neutral second check on held-out problems;
   - 25 passes: one attention head switched off (layer 15, head 7), +8.3 points;
   - 50 passes: 64 MLP columns in layers 15 and 19, +49.8 points.

   On MATH-500 they gave 12.3 / 14.1 / 48.1. There is one run per budget, so we cannot separate the effect of the
   budget from run-to-run variation. IFEval does not even move in one direction (49.0 at 10 passes, 39.9 at 25).
5. **There is no cheap baseline that leaves the weights alone.** The base and edited models were compared under only
   one setup, which is prone to loops (thinking mode on, 32,768 tokens). In "Response to Comments", the authors
   decline the thinking-off comparison because "it avoids the loops by not using the mode that has them". That is
   fair for the narrow claim that the loop mechanism can be found and edited. It is not fair for the practical claims
   ("better at math", "raising IFEval by 3x"). For scale, Qwen's model card reports IFEval 52.1 for Qwen3.5-0.8B with
   thinking off (https://huggingface.co/Qwen/Qwen3.5-0.8B), against the blog's thinking-mode base of 21.0 and edited
   score of 62.3. The setups differ, so this comparison only gives a rough idea. Sampling settings were not explored
   either. The card says presence_penalty can be raised (0 to 2) "to reduce endless repetitions" (README line 940),
   and its warning about loops (line 856) refers to the recommended thinking-mode settings. Neither a sampling change
   nor a token limit was tried as a baseline.
6. **There is no standard baseline for the edit itself.** There is no automatic difference-in-means direction
   removal (the Arditi et al. recipe, run without an LLM agent), and no small fine-tune with the same number of
   forward passes. So we cannot tell how much of the success comes from the agent's reasoning and how much from
   running a standard recipe. The only random control is a single random direction, tried once in the middle of the
   search (+1.6 points, edit-log entries 11 and 12). It was not repeated on the final edit or on the benchmarks.
7. **"0.016% of weights" does not show how small the change is.** The edit is rank-one in two matrices, at strength
   6. With W ← W − α r rᵀ W, the part of the output along r is multiplied by 1 − α, so it is reversed and made about
   5x larger, not removed. The 0.016% is also a share of a total that includes the 254M-entry tied embedding. Damage
   elsewhere has to be measured. Counting touched entries does not show it.
8. **Damage was checked only where removing loops helps.** All eight benchmarks used the same setup (thinking mode,
   32,768 tokens), so they mostly measure whether the model finishes or not. The base model's GPQA Diamond score is
   22.7%, below the 25% you would get by guessing on a 4-option test. That fits many unanswered questions (our
   inference). No damage check is reported where loops play no part, such as perplexity on held-out text, accuracy
   with thinking off, or other languages. The agent's own report notes that "some degeneration remains".
9. **Small numbers, and unclear numbers of attempts, in Experiments 1, 2, 4 and 5.** There are 4 game networks (each
   the best of 3 PPO seeds), 3 diffusion models, 1 car level and 3 games. There are no chance baselines, and failed
   attempts are not reported, except in Experiment 7 and for Gemma. In Experiments 1 and 2, the input and output
   layers map onto labelled spatial grids, so reading them is close to reading an image. The authors admit this at
   the start of Experiment 3.
10. **Experiments 4 to 6 test writing a hand-designed program into weights, not understanding a trained network.**
    - In Experiment 4, the prompt asks to "fix only its cause". But the edit takes over a few units per layer (units
      122 to 127 of the first actor layer, fewer later) to build "a small separate path". That path scripts a
      reverse-then-launch move and then hands control back. It goes around the PPO policy's flaw instead of
      repairing it.
    - In Experiment 5, the agent had the game's source code, while PPO learns only from experience, as the authors
      admit ("puts a price on that in experience").
    - In Experiment 6, with forward passes, the agent tunes a Go heuristic with settings (the code shown has a
      `DEFAULTS` dict, for example `capture=9.0`) and "keeps a change only if it wins more".
11. **Experiment 7 works against the general claim for real language models.** On pairs of Gemma 3 270M models,
    reading the weights alone failed in all four attempts. Success needed activations from running the models. This
    is the post's only clean weights-only test on a language model, and it failed.
12. **The "not a method" framing.** "Response to Comments" says Weight-Space Programming "is not a method", and that
    the baselines "are not there to rank methods". Yet the summary at the top leads with comparisons (beating PPO at
    100M steps; 7.9% to 48.1%; IFEval 3x). For an organisation deciding what to invest in, comparison with the
    alternatives is exactly the question. We did not check the list of outside replications (unverified), and we do
    not rely on it.

## 4. Which conclusions hold, and which do not

**These hold, with the caveats above:**
- A frontier LLM with Python can recover known facts about circuits (copying heads) from the weights of tiny
  transformers, without running them (Experiment 3).
- With access to activations and a budget of forward passes, an LLM agent can find a low-rank edit that stops
  Qwen3.5-0.8B's thinking loops under the blog's thinking-mode setup, and the gain holds on benchmarks it never saw
  (Experiment 8).
- That gain is mainly unblocking. The rescued answers had almost all been cut off before, and the same procedure did
  not improve a model that rarely loops (Experiment 8 and Gemma).
- LLMs can write hand-designed policies into small networks from a description (Experiment 5), and into larger ones
  with search (Experiment 6).

**These do not hold:**
- That the Experiment 8 edit came from reading the weights, or was "native". It was located with activations, its
  strength was tuned with feedback, and the prompt suggested the failure to look for.
- That Qwen became "better at math", or better at following instructions, as a skill. The data show it stopped
  looping in a mode where the base model rarely finished.
- That the edit beats simple alternatives (thinking off, a sampling change, a standard direction removal, a small
  fine-tune). None of these were tested.
- That changing 0.016% of the weights means little damage elsewhere. Damage was not measured outside the loop
  setting.
- That the result is reliable across runs. There is one run per budget, and the three runs found three different
  mechanisms.
- That LLMs can "natively read" the weights of realistic language models. Experiment 7 failed with weights alone.
- The "What Comes Next" ideas (auditing without evaluation, adding skills with no data, avoiding catastrophic
  forgetting, recursive self-improvement). These are speculation; no experiment in the post tests them.

## 5. What this means for SE.LLMA

The useful idea is narrow. A cheap, low-rank edit found from activations can sometimes remove a behavioural blocker
that hides an ability the model already has. The blog does not show whether that beats prompting, sampling or
fine-tuning, or what it costs elsewhere. A Swedish example of something like thinking loops is a reply in the wrong
language (English, Norwegian or Danish) to a Swedish prompt. Our experiment tests whether the Experiment 8 recipe (a
difference-in-means direction, removed along that one direction in a few layers, with the strength chosen using
development feedback) removes that blocker better than random edits of the same size, a system prompt, and the
sampling settings from the model card. We count our development runs as the feedback budget, and we call our method
a feedback-guided, activation-derived weight edit, not weight reading.
