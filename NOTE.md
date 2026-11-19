# Can a small weight edit stop wrong-language replies to Swedish prompts?

Research note for the SE.LLMA take-home. Code, data and every output are in this repository. Each number here comes
from a file in `results/` (see the appendix), except the blog's own numbers in section 1, which are linked.

**What this tests.** We tested the kind of edit the blog's Experiment 8 used: a feedback-guided, activation-derived,
low-rank edit of the matrices that write into the residual stream. We did not test whether an AI can find edits by
reading weights. Our edit was derived from the model's activations and chosen with dev feedback.

## 1. Critique of the blog

"Sparks of Weight-Space Programming" ([blog](https://rikiyat.github.io/blog/weight-space-programming/)) says an LLM
agent can read and edit neural-network weights. Its experiments range from pure weight reading to edits optimized
with model feedback. Full critique: `results/critique_own.md`; comparison with the researcher's notes:
`results/critique_comparison.md`.

- **Strongest evidence.**
  - Exp. 3: from the weights of tiny character models, without running them, the agent named exactly the copying
    heads that a separate measurement found.
  - Exp. 8: the frozen Qwen3.5-0.8B edit, tested on 1,152 held-out practice problems, gained +49.8 points (SE 1.5),
    with 579 rescues against 5 losses.
  - Exp. 8 shows unblocking, not new skill: 576 of the 579 rescues had previously been cut off at the token limit,
    and the same procedure did not help Gemma 3 270M, a model that rarely loops.
- **Exp. 8 is not weight reading.**
  - The loops were found from generated text and localized with activations returned by forward passes.
  - The strength was raised step by step on measured accuracy (1, 2, 3, 4, then 6 on 64 neurons).
  - The task prompt itself named repetition as an example failure to fix.
  - The post's one weights-only test on a language model (Exp. 7) failed in all four attempts.
- **Missing baselines.**
  - There is no thinking-off or sampling baseline for the practical claims ("better at math").
  - There is no automated direction removal without an agent, and no small fine-tune.
  - The random-direction control was a single draw. The logs show +1.56 points (SE 3.52) on 128 generations, which
    is enough at that effect size, but it is one draw.
  - The three feedback budgets produced three different edits, with one run each.
  - "0.016% of weights" counts touched entries, not the size of the change: strength 6 reverses and amplifies the
    component along the direction.
- **Justified:** an activation-derived edit, tuned with feedback, can remove a behavioural blocker that the model's
  own outputs reveal. **Not justified:**
  - that LLMs can read the weights of realistic language models;
  - that Qwen became better at math as a skill;
  - that the edit beats simple alternatives.

## 2. Hypothesis

Question: can a small targeted weight edit stop a small multilingual model from replying in the wrong language
(English, Norwegian or Danish) to Swedish prompts, better than simple alternatives and without collateral damage?
The researcher's bet (unverified going in): wrong-language replies are a blocker, not a missing skill, like
Qwen's thinking loops in the blog.

- **Supported if:** on held-out test prompts, the edit beats both the base model and random edits of the same size,
  regressions are small, and no prompt or sampling change matches it.
- **Challenged if:** random edits do as well, a prompt or sampling change does as well or better, other abilities
  get worse, or the failure barely exists.

## 3. Setup

- **Model.** Qwen2.5-0.5B-Instruct (revision `7ae5576`, Apache-2.0), fp32 on Apple MPS.
  - The planned model, Qwen3.5-0.8B (non-thinking default template), gave no wrong-language reply to the 40 pilot
    prompts in the first 128 tokens: 0 of 40, Wilson 95% CI 0–8.8% (greedy decoding, no system prompt, whole-reply
    label). That says nothing about longer replies.
  - Qwen2.5 gave 6 of 40, so it became the model, as the plan's pilot rule says (switch below 10%).
- **Data** (all prompts and questions written by Claude; `data/`):
  - 300 everyday Swedish prompts in 15 categories, split 150 dev / 150 test, stratified, seed 0;
  - 100 English test prompts and 50 English dev prompts;
  - 60 exact-answer questions (30 Swedish, 30 English);
  - 100 Wikipedia paragraphs (50 Swedish, 50 English) for perplexity;
  - 200 more paragraphs (Swedish, English, Norwegian, Danish) to check language identification and to compute the
    Swedish-minus-English direction.
- **Language identification.**
  - lingua over all languages: 100% correct on 200 whole paragraphs, 93.5% on their first sentences.
  - A reply counts as Swedish only if both of its halves are classified Swedish; replies under 40 words are judged
    whole. The same rule defines English replies.
  - This rule was fixed before any 512-token run.
- **Decoding.** Greedy (repetition penalty 1.0), the model's default chat template (which adds an English default
  system prompt), up to 512 new tokens (64 for the questions), batch 32.
- **The edit.**
  - W ← W − α r rᵀ W on the attention-output and MLP down-projection matrices of four consecutive layers.
  - α = 1 removes what these matrices write along the unit direction r, α = 2 flips it, and α = −1 and −2 multiply
    it by 2 and 3. r and −r give the same edit.
  - Free parameters: one 896-number direction and one strength, plus the choice of layer window.
- **Dev work (the feedback).**
  - Round 1 (the plan, 128 tokens): the base model failed on 18 of 150 dev prompts, fewer than the plan's 20, so the
    direction was the plan's fallback: Swedish minus English Wikipedia text.
  - Removing or flipping it in four layer windows (8 runs) never beat the base model by more than one prompt.
  - Removing it in layers 16–19 made almost every reply non-Swedish. These layers' output along the direction is
    needed for Swedish replies, and removing or flipping it cannot add more of it.
  - Round 2 (512 tokens, the researcher's last design change) had eight settings:
    - the Swedish-minus-English component multiplied (α −1, −2);
    - the failure direction (wrong-language minus Swedish dev replies) removed or flipped (α 1, 2);
    - each in layers 12–15 and in layers 16–19.
  - Selection:
    - First filter: a setting's Swedish replies must have a base-model perplexity within 1.15× of the base model's
      own replies.
    - Then rank by dev Swedish-reply rate.
    - Take the first ranked setting whose English-reply rate on the 50 English dev prompts is within 5 points of the
      base model's.
- **Test conditions** (each run once):
  - base;
  - edited;
  - 5 random unit directions with the same layers, sign and per-matrix size of change (Frobenius norm);
  - system prompt "Svara alltid på svenska.";
  - empty system prompt (it replaces the English default);
  - sampling with the repo's `generation_config.json` (temperature 0.7, top-p 0.8, top-k 20, repetition penalty
    1.1), 3 seeds;
  - LoRA (rank 8, 3 epochs), trained only on dev prompts: the base model's own Swedish replies under the Swedish
    system prompt, trained without that prompt. Its own dev evaluation was not run: the process stalled with memory
    full after training, and its summary was written by hand from the logs.
- **Metrics.**
  - Target: Swedish-reply rate on Swedish test prompts.
  - Regressions: English-reply rate on English prompts, exact-answer accuracy, Wikipedia perplexity.
  - Not gamed: base-model perplexity of each condition's Swedish replies, flagged above 1.15× base.
  - Statistics: paired comparisons on the same prompts, with exact McNemar tests, paired-bootstrap 95% CIs (10,000
    resamples) and the minimum detectable difference.
- **Test-set use.**
  - An early smoke test read the first 10 items of every test file and showed aggregate results on them, before
    the final design was fixed.
  - The main analysis therefore uses only the untouched items: 140 Swedish, 90 English, 50 questions and 90
    paragraphs. Full-set numbers are secondary.

## 4. Results

## 5. Interpretation: unblocking or new skill?

## 6. Limitations

## 7. Recommendation and next experiment

## Appendix: where each number comes from
