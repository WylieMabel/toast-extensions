# The Token-Mixing Translator

Design note. Nothing here is implemented yet — this scopes the work and states the argument.

## The problem, plainly

A ViT does not see an image as a picture. It chops it into a grid of square patches and turns
each patch into a vector. For DINOv2-base at 224px that's a 16×16 grid = 256 patches, plus one
extra "CLS" summary token, so 257 tokens, each 768 numbers long. A hidden state is therefore a
block of numbers shaped `[batch, 257, 768]`.

When TOAST deletes some blocks, it learns a **translator**: a small map that takes the hidden
state where the deleted blocks started and produces something close to what they would have
output. Today that map is built like this ([module.py:751-752](../src/toast/modules/module.py#L751-L752)):

```python
x_flat = x.reshape(-1, x.shape[-1])   # [B, 257, 768] -> [B*257, 768]
```

That line tips every token from every image into one big pile and fits **a single 768×768
matrix**, which is then applied to each token **separately**. Token 5 gets transformed without
ever seeing token 6. Nothing moves sideways across the image.

## Why that is fatal for attention specifically

For an **MLP sub-block** this is fine, because an MLP genuinely does process each token on its
own. Approximating it with a per-token map is the right shape of tool.

For a **self-attention sub-block** it is not a matter of degree. Moving information between
tokens is the entire operation attention performs — that is what "attention" *means* here. A map
that provably cannot move information between tokens cannot represent it. Not "approximates it
poorly": cannot express it at all, at any width, with any amount of fitting data.

This is the strongest form of argument available: structural, not empirical. The experiment
doesn't discover the limitation, it *measures the cost* of a limitation already known from the
form of the equations.

## What a token mixer is

A translator that is also allowed to move information across positions. Two candidates:

**Dense N×N mixer** (MLP-Mixer style). A 257×257 matrix letting every token read every other
token. At 257 tokens that's ~66k parameters — cheap. Downside: it is locked to one input
resolution, since the matrix size *is* the token count.

**Depthwise convolution.** Fold the 256 patch tokens back into their 16×16 spatial grid, slide a
small 3×3 or 5×5 kernel across it so each token mixes with its spatial neighbours, then a
channel-mixing MLP. Far fewer parameters, resolution-flexible, and it builds in the prior that
nearby patches are the relevant ones — which is true of images and not true of arbitrary
sequences.

**Hypothesis:** on isolated attention blocks, the per-token linear translator loses accuracy
that a depthwise-conv translator recovers. On MLP blocks, both should perform the same — and
that null result matters just as much, because it shows the gain is specifically about token
mixing rather than the mixer simply having more parameters.

## What implementing it actually requires

**1. A shape-preserving mode.** `fit_translators` and `transform_similar_spaces`
([module.py:739-787](../src/toast/modules/module.py#L739-L787)) support mode 1 (flatten all
tokens together) and mode 2 (one separate map per token position). *Both flatten.* Neither can
carry a mixer, so a `mode=3` that keeps `[B, N, D]` intact is a prerequisite for everything else.

**2. The CLS token is the fiddly part.** Token count is `N = 1 + P²`. The conv needs a square
grid, but CLS has no position in it — it is a summary vector, not a patch. So the translator has
to split CLS off, reshape the remaining 256 into 16×16, convolve, flatten back, and re-attach
CLS with its own separate per-token linear map. Getting this wrong silently scrambles the
spatial layout, which is the kind of bug that produces plausible-but-wrong numbers. Worth an
explicit test that a known pattern survives the round trip.

**3. `conv_translator.py` is not a starting point.**
[src/toast/modules/conv_translator.py](../src/toast/modules/conv_translator.py) exists but is
dimensionally broken — hardcoded 128/384 channels unrelated to any actual model width, and a
`Conv1d` wedged between two `Linear` layers where the shapes cannot line up. It is also not
registered in `NAME2TRANSLATORS`, so it has never run. Rewrite it rather than patching it.

**4. The attention-skipping evaluation needs one more change than it looks.**
`AttentionLinearisedEncoder.fit` ([module.py:327-393](../src/toast/modules/module.py#L327-L393))
does **not** route through `NAME2TRANSLATORS` at all — it runs its own `lstsq` internally and
concatenates the hooked activations down to `[N·seq, d]`. So testing a mixer on attention
sub-blocks needs both the shape-preserving treatment *and* a translator-name parameter threaded
into that class. This is the part most likely to be underestimated.

**Cheaper fallback that still tests the hypothesis:** compare `dwconv` against `linear` as the
**whole-block skip bridge** instead. That path *does* route through `NAME2TRANSLATORS`, so it
needs no change to `AttentionLinearisedEncoder` at all. A skipped whole block contains both
attention and MLP, so the test is less clean — but it is a day of work rather than three, and a
positive result there justifies the fuller version.

## Suggested order

1. `mode=3` plus a shape round-trip test (no fitting yet — just confirm `[B,N,D]` survives).
2. Dense N×N mixer. Simpler, no CLS/grid reshaping, so it isolates "does token mixing help?"
   from "is my grid handling correct?".
3. Depthwise conv, once the dense version has confirmed the effect exists.
4. Only then extend `AttentionLinearisedEncoder` for the isolated-sub-block version.

Doing the dense mixer first is the important bit: if it shows no gain over per-token linear,
the depthwise-conv work is not worth starting, and that is worth knowing after step 2 rather
than after step 4.
