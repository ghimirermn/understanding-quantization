 > **why quantization exists at all**

first question is not how. first question is why.

models are trained in fp32. training requires precision because gradients are small, updates are subtle, and instability compounds.

but inference is different.

during inference:
- no gradients
- no accumulation over steps  
- just forward pass
- mostly matrix multiplications
- mostly memory movement

modern accelerators are memory bandwidth bound, not compute bound.

moving 32-bit floats is expensive:
- 4× more memory than int8
- more cache misses
- higher energy
- slower throughput

so the real reason quantization exists is: **we trade representational precision for memory efficiency, latency, and hardware compatibility.**

it is a systems problem before it is a math problem.

---

> **what quantization actually is**

quantization is mapping a continuous (or high precision discrete) space into a lower precision discrete lattice.

fp32 → int8, fp16 → int4, etc.

formally: given a real value x ∈ [α, β], we map it into a discrete integer space:

x_q = round(x / s) + z

where:
- s = scale
- z = zero point (optional)

dequantization: x ≈ s(x_q - z)

there will always be error. the question is whether that error matters.

---

> **symmetric vs asymmetric**

**symmetric:**
- zero-point = 0
- range like [-127, 127]
- simpler hardware math
- usually used for weights (since weights are centered)

**asymmetric:**
- has non-zero zero-point
- range like [0, 255]  
- better for activations (especially relu)

but the deeper issue: when doing matmul, (w_q - z_pw)(x_q - z_px) expands into correction terms. that increases kernel complexity.

so symmetric is often preferred for weights because it simplifies integer gemm kernels.



---

> **the real problem: choosing the range [α, β]**

everything depends on this.

- too small → clipping → bias
- too large → wasted precision → variance

common strategies:
- min–max
- percentile clipping
- mse minimization
- kl divergence
- learned scale (lsq — learned step size)
- hessian-aware methods
- gptq / awq (for llms)

this is not just grid search.

some layers are extremely sensitive. others are robust.

so the deeper question becomes: **where can we afford error?**

---

> **layer-wise vs channel-wise**

**layer-wise:**
- one scale per layer
- simple
- but sensitive to outliers

**channel-wise:**
- separate scale per output channel
- better accuracy
- more scale storage
- slightly more kernel complexity

intuitively: if one channel has large magnitude, it stretches the whole range in layer-wise quantization. result: precision collapse for other channels.

so per-channel reduces representational bias.

again: accuracy vs complexity tradeoff.

---

> **post-training quantization (ptq)**

no retraining. just calibrate on small dataset.

assumes:
- model is overparameterized
- small perturbations don't move decision boundary much

often works surprisingly well.

**fails when:**
- activations are heavy-tailed
- attention distributions are sharp
- model operates near decision boundary

ptq works because modern models are redundant — and because batch normalization and weight decay already regularize weights toward zero-mean, unit-variance distributions that quantize cleanly.

---

> **quantization-aware training (qat)**

inject fake quantization during training.

forward pass simulates rounding. backward pass uses straight-through estimator (gradients flow through round as if identity).

the network learns to live inside the quantized lattice.

this is deeper than "retraining." it reshapes the optimization landscape under precision constraints. the model adapts its geometry — exploiting gradient noise coherence where the bias correlates with directions that hurt quantization.

---

> **why 8-bit works at all**

this is the more interesting question. why can we destroy 75% of precision and still get almost same accuracy?

possible reasons:
- overparameterization
- flat minima
- low intrinsic rank of representations (trained networks live on low-dimensional manifolds; quantization coarse-discretizes along null directions)
- redundancy in learned features
- decision boundaries are robust to small perturbations

quantization works because models are not using full 32-bit precision meaningfully.

connect to hessian: eigendecomposition of loss landscape tells you which directions are sensitive. quantization noise in high-curvature directions hurts; in flat directions, it's invisible. this is why hessian-aware methods outperform uniform schemes.

---

> **information perspective**

quantization reduces entropy. we compress representational capacity.

so the question becomes: **how many bits are needed to preserve decision boundaries?**

this connects to:
- rate–distortion theory
- compression theory
- redundancy in deep networks

---

> **transformers and modern relevance**

for convnets, quantization is mostly solved.

for transformers:
- weight memory is huge
- kv cache explodes during generation
- inference is memory dominated

weight-only 4-bit quantization is now standard in llm inference — specifically post-training with grouping (gptq, awq).

quantization is no longer optional here. 

**the twist:** transformers are *more* quantizable than convnets in weights (layer norm, residual streams), but *less* quantizable in activations (attention outliers). this flips the problem structure. kv-cache quantization is harder due to auto-regressive dependencies — hence why vllm uses fp8 for kv in some kernels, not int8.

---

> **the hardware contract**

what do nvidia tensor cores, arm neon, and apple ane actually require?

symmetric, per-channel, power-of-two scales are preferred ?

also: numerical overflow in integer accumulation, handling of bias vectors, fusion of quantization with convolution kernels. the implementation pathology matters.

---

> **open deeper questions**

- why are some layers more sensitive?
- how does quantization noise propagate?
- is there a theoretical lower bound on bits?
- how does hessian spectrum relate to quantization robustness?
- can quantization be viewed as structured noise injection?
- why does per-token dynamic quantization work for activations but not weights?

weights are static, optimize representation once. activations have dynamic range. quantization error for weights is systematic (biased); for activations it's stochastic (averages out across tokens).
