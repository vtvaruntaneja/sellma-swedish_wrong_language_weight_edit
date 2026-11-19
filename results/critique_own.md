# Critique of "Sparks of Weight-Space Programming" (own reading, written before seeing my_notes.md)

Source: Takehi, Teoh, Zhang, Khattab, blog post dated 2 Nov 2026, "Response to Comments" added 12 Nov 2026
(https://rikiyat.github.io/blog/weight-space-programming/). Experiment 8 details come from the blog's own data files:
the edit logs (https://rikiyat.github.io/blog/weight-space-programming/assets/mathedit/trajectories.json),
the agent prompt and final answers (https://rikiyat.github.io/blog/weight-space-programming/assets/traces/exp8_math.json)
and the benchmark points (https://rikiyat.github.io/blog/weight-space-programming/assets/mathedit/benchmarks.json).
All numbers below are numbers the blog reports. None are ours.

## 1. The key distinction: what the editing agent could see

The assignment asks us to separate reasoning about weights from optimizing edits with repeated model feedback.
The blog's experiments sit at very different points on that scale:

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

Only Experiments 1 to 3 (and 5, which is writing a program from a spec) are reasoning without running the model.
The two results that matter most for real language models (7 and 8) depend on running the model.

## 2. Strongest evidence

1. **Experiment 3, copying heads.** From four transformer blocks of a 7.2M-parameter character model, with Python but no
   forward passes ("We checked every run"), the agent named exactly the induction ("copying") heads that the authors
   measured independently by running the model, in all three models, and named no other head. Ground truth is
   measured by a separate procedure, so this is a real test. Caveats: the method it used is a known one (weight
   composition, Elhage et al. 2021, which the blog points out), the models are tiny, and the language question was a
   forced choice among three options given in the prompt, answered with confidence 0.48 to 0.62.
2. **Experiment 8, held-out confirmation.** The 50-pass edit was frozen and then tested on 1,152 practice problems not
   used to choose it: +49.8 points (SE 1.5), 579 wrong-to-right against 5 right-to-wrong (agent's final report in
   `trajectories.json`). MATH-500, which the agent never saw, went from 7.9% to 48.1%. The effect of the edit on
   looping under that protocol is real and large.
3. **The mechanism is mostly "unblocking", and the authors say so.** 576 of the 579 rescues were answers that had
   previously been cut off at the token limit, and accuracy on answers that already finished was unchanged (agent's
   final report). In the "Additional experiment", the same procedure could not improve Gemma 3 270M, a model that
   rarely loops (MATH-500 6.8% base, 7.0 to 7.7% edited). Together these are the best evidence in the post for the
   distinction between removing a blocker and adding a skill. It is one non-looping model, of a different size and
   family, so it is suggestive, not conclusive.
4. **Transparency.** The logs, agent prompt, failures (Exp. 7 weights-only, Gemma) and the edited model are public
   (https://github.com/RikiyaT/lm_understand-logs, https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
   This makes the points below checkable.

## 3. Most important limitations

1. **The TL;DR describes Experiment 8 as weight reading; the log shows feedback-guided search.** The TL;DR says the
   agent "identifies an issue in its weights ... and fixes this by natively editing 0.016% of the weights". In the log,
   the issue was found from generated text (turn 3: "85/96 outputs were cut off"). The site was then localized with
   activations returned by forward passes ("Matched productive/repetitive contexts localized the relevant MLP
   writers"). The final edit is a direction removal of the Arditi et al. (2024) kind, which the blog itself cites.
   This is activation-derived and feedback-tuned, not read off the weights.
2. **The prompt gave the hypothesis and the method.** The Experiment 8 task prompt (`exp8_math.json`) says: "for
   example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives
   that repetition counts". Its "HOW TO WORK" section also prescribes the protocol (diagnose first, localize
   causally, try several strengths for a dose-response, use paired held-out tests). The agent carried out a search
   the experimenters had outlined. It did not discover the failure mode from the weights.
3. **The strength and the components were tuned on model feedback.** In the 50-pass log the strength went
   1 → 2 → 3 → 4, then to a 64-neuron subset at strength 6 (edit-log entries 04 to 17), each step after measuring
   accuracy. About the first 33 passes went to diagnosis and tuning and the rest to frozen confirmation (turn-by-turn
   `used` counter in `trajectories.json`). This is hill-climbing with an LLM as the search policy.
4. **The result depends on the budget and on the run, and the "issue found" is not stable.** The three budgets gave
   three different edits. With 10 passes: 30 rank-one edits in layers 8 to 22, +3.7 points on practice, judged "modest"
   by the agent, with a neutral second holdout. With 25 passes: one zeroed attention head (layer 15, head 7), +8.3
   points. With 50 passes: 64 MLP columns in layers 15 and 19, +49.8 points. MATH-500: 12.3 / 14.1 / 48.1. There is one
   run per budget, so budget effects cannot be separated from run-to-run variance. IFEval is non-monotonic
   (49.0 at 10 passes, 39.9 at 25).
5. **No cheap non-weight baseline.** Base and edited models were compared only under one loop-prone protocol
   (thinking on, 32,768 tokens). The "Response to Comments" declines the thinking-off comparison because "it avoids
   the loops by not using the mode that has them". That is fair for the narrow claim that the loop mechanism can be
   found and edited. It is not fair for the practical claims ("better at math", "raising IFEval by 3x"). For scale,
   Qwen's model card reports IFEval 52.1 for Qwen3.5-0.8B in non-thinking mode
   (https://huggingface.co/Qwen/Qwen3.5-0.8B). The blog's thinking-mode base is 21.0 and its edited model 62.3.
   The protocols differ, so this comparison is only indicative. Sampling was also not explored: the card says
   presence_penalty can be raised (0 to 2) "to reduce endless repetitions" (README line 940), and its loop warning
   (line 856) is about the recommended thinking-mode settings. Neither a sampling change nor a token budget was tried
   as a baseline.
6. **No conventional baseline for the edit itself.** There is no automated difference-in-means direction removal
   (Arditi et al., no LLM agent) and no small fine-tune with the same forward-pass budget. So we cannot tell how
   much of the success comes from the agent's reasoning and how much from executing a standard recipe. The only
   random control is a single random direction, run once mid-search (+1.6 points, edit-log entries 11 and 12). It was
   not re-run on the final edit or on the benchmarks.
7. **"0.016% of weights" is not a measure of how small the change is.** The edit is rank-one in two matrices, at
   strength 6. With W ← W − α r rᵀ W, the component along r is scaled by 1 − α, so it is reversed and amplified about
   5x, not removed. The denominator also includes the 254M-entry tied embedding. Collateral damage has to be
   measured. Counting touched entries does not show it.
8. **Regressions were checked only where loop removal helps.** All eight benchmarks were run under the same
   thinking-mode, 32,768-token protocol, so they mostly measure finishing versus not finishing. Base GPQA Diamond is
   22.7%, below the 25% chance rate of a 4-option test, which is consistent with many unanswered questions (our
   inference). No regression measure is reported where loops play no role, such as held-out perplexity,
   non-thinking-mode accuracy or other languages. The agent's own report notes that "some degeneration remains".
9. **Small n, and unclear attempt counts, in Experiments 1, 2, 4 and 5.** There are 4 game networks (each the best of
   3 PPO seeds), 3 diffusion models, 1 car level and 3 games, with no chance baselines and no account of failed
   attempts outside Exp. 7 and Gemma. In Exp. 1 and 2 the input and output layers map onto labelled spatial grids, so
   reading them is close to reading an image. The authors acknowledge this at the start of Exp. 3.
10. **Experiments 4 to 6 test compiling a hand-designed program into weights, not understanding a trained network.**
    In Exp. 4 the prompt asks to "fix only its cause", but the edit repurposes a few units per layer (units
    122 to 127 of the first actor layer, fewer later) into "a small separate path" that scripts a reverse-then-launch
    manoeuvre and then hands control back. It bypasses the PPO
    policy's flaw rather than repairing it. In Exp. 5 the agent had the game source code, while PPO learns from
    experience only, as the authors concede ("puts a price on that in experience"). In Exp. 6, with forward passes,
    the agent tunes a parameterized Go heuristic (the shown code has a `DEFAULTS` dict such as `capture=9.0`) and
    "keeps a change only if it wins more".
11. **Experiment 7 cuts against the general claim for real LMs.** On Gemma 3 270M pairs, reading weights alone failed
    in all four attempts. Success needed activations from running the models. This is the post's only clean
    weights-only test on a language model, and it is negative.
12. **The "not a method" framing.** "Response to Comments" says Weight-Space Programming "is not a method" and that the
    baselines "are not there to rank methods". Yet the TL;DR leads with comparative numbers (beating PPO at 100M
    steps; 7.9% → 48.1%; IFEval 3x). For an organisation deciding what to invest in, comparison with alternatives is
    exactly the question. The list of external replications was not checked by us (unverified) and we do not rely
    on it.

## 4. Which conclusions are and are not justified

**Justified (with the caveats above)**
- A frontier LLM with Python can recover known circuit facts (copying heads) from the weights of tiny transformers
  without running them (Exp. 3).
- With activation access and a forward-pass budget, an LLM agent can find a low-rank edit that stops Qwen3.5-0.8B's
  thinking loops under the blog's thinking-mode protocol, and the gain holds on benchmarks it never saw (Exp. 8).
- That gain is mainly unblocking: rescued answers were almost all previously truncated, and the same procedure did
  not improve a model that rarely loops (Exp. 8 and Gemma).
- LLMs can write hand-designed policies into small networks from a specification (Exp. 5), and with search into
  larger ones (Exp. 6).

**Not justified**
- That the Exp. 8 edit came from reading the weights, or was "native": it was activation-localized, strength-tuned on
  feedback, and the target failure was suggested in the prompt.
- That Qwen became "better at math", or better at instruction following, as a capability. The data show it stopped
  looping in a mode where the base model rarely finished.
- That this beats simple alternatives (thinking off, sampling changes, a standard direction removal, a small
  fine-tune). None were tested.
- That a change to 0.016% of weights implies little collateral damage. This was not measured outside the loop setting.
- That the result is reliable across runs: there is one run per budget, and the three runs found three different
  mechanisms.
- That LLMs can "natively read" weights of realistic language models. Exp. 7 failed weights-only.
- The "What Comes Next" items (auditing without evaluation, zero-data capability addition, avoiding catastrophic
  forgetting, recursive self-improvement). These are speculation; no experiment in the post tests them.

## 5. What this means for SE.LLMA

The usable idea is narrow: a cheap, low-rank, activation-derived edit can sometimes remove a behavioural blocker that
hides an ability the model already has. Whether that beats prompting, sampling or fine-tuning, and what it costs
elsewhere, the blog does not show. A Swedish-relevant analogue of thinking loops is replying in the wrong language
(English, Norwegian or Danish) to a Swedish prompt. Our experiment tests whether the Exp. 8 recipe (difference-in-means
direction, rank-one removal in a few layers, strength chosen on dev feedback) removes that blocker better than random
edits of the same size, a system prompt, and the card's sampling settings. We count our dev evaluations as the
feedback budget and call our method a feedback-guided, activation-derived weight edit, not weight reading.
