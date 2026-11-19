# My notes on the blog

My own reading of "Sparks of Weight-Space Programming", 18 Nov 2026. Verify each point against the blog and tell me where I am wrong.

## Strongest evidence
- **Exp 3:** from the four middle blocks only, and without running the network, Astra named the exact copying (induction) heads in all three tiny language models. They matched the heads measured by running the models.
- **Exp 4:** in one reply with no tools, Astra fixed the failing Kinetix car policy: 0% success became 100% over 1,024 episodes, with 3.5% of the weights changed.
  - Caveat: it was not weights alone. Astra also got a scene description and one recorded failing episode.
- **Exp 8:** Astra's own run included checks:
  - a random direction of the same size, which did almost nothing (+1.6 points on practice problems; Experiment 8 edit log, turns 88-92)
  - layer-isolation tests

## Weak points
- **Small samples:** 4 games, 3 shapes, 3 tiny language models, 3 Gemma pairs. Only one editing agent (GPT-6 Astra) was tested, though it worked on several target networks.
- **Hints in the prompts:**
  - Exp 1 described the input channels (for example "agent", "reward targets").
  - Exp 3's prompt listed the three answer options.
- **Reading weights alone failed on a real language model.** In Exp 7 it failed all 4 attempts, and only succeeded once given an attention tool that requires running the models.
- **Exp 8 is feedback-guided, not reasoning from weights alone:**
  - It used up to 50 forward passes on practice problems (GSM8K and MATH training sets, not the held-out benchmarks).
  - The edit log shows the edit strength tried at 1, 2, 3 and 4, with a final of 6, each kept by measured gains.
  - With 10 or 25 forward passes, looping barely changed: 87.2% and 86.4% of MATH-500 answers still hit the token limit.
- **Exp 8's gain is unblocking, not new skill:**
  - The base model hit the token limit on 85 of its first 96 practice answers.
  - Astra's own summary says 576 of the 579 rescued answers had originally been cut off.
  - Gemma 3 270M, which rarely loops, showed no meaningful gain.
- **A missing comparison, not proof the result is wrong.** My reading of the Qwen3.5-0.8B model card, which needs verifying:
  - The 0.8B model is more prone to thinking loops.
  - It runs in non-thinking mode by default.
  - It recommends temperature 1.0 for thinking mode on text, while the blog used thinking mode at 0.6.
  - The blog says turning thinking off raises the base score, but gives no number.
- **It is a blog post, not a peer-reviewed paper.**
