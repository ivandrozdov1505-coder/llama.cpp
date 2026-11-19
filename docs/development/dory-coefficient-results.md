# Dory coefficient precomputation experiment

I tested a Dory decode optimization on my RTX 2060 6 GB. The graph used to recalculate casts and scalar transforms of fixed weight vectors for every decoded token and recurrent-layer visit. I now prepare these vectors once per working graph and reuse them on the same device as each layer's matrix weights.

The change is limited to `src/models/dory.cpp`:

- Prepare F32 coefficients for residual alpha, FFN gate/up scales, attention QK/gate scales, and output scaling.
- Place each coefficient buffer alongside the corresponding layer matrix. Keeping prepared F32 vectors in the original BF16 vector's host buffer was slower in my tests.
- Keep buffers owned by the graph and release them when it is reset.
- Use the ordinary graph operations during the memory-fit pass, when weights have not been allocated yet.
- Reuse coefficients across visits to the same physical recurrent layer. The final version also uses explicit SwiGLU; testing that change alone did not show a gain.

I did not reduce the model's loop count, context, quantization precision, or KV-cache precision.

## Measurements

These are two separate staged A/B experiments, not an end-to-end stock-versus-final comparison. The intermediate shared-node build appears in both runs, and its speed varied with background load. The percentages should not be added or multiplied to claim a cumulative gain.

Base revision: `98a74c97ea09ac72f6ceda1559808fc0a2f447ce` from the Dory support branch. Windows x64, CUDA Release build for sm_75, CUDA graphs enabled, Flash Attention disabled. NVIDIA's Dory Q5_K_M GGUF, 2 loops, context 8192, last_loop, K q8_0 / V f16, 99 GPU layers, batch 256 / micro batch 128, 5 CPU threads. Requests used temperature 0, seed 42, ignore_eos, cache_prompt false, and exactly 128 generated tokens. Each model load had a 32-token warmup.

The load order was A-B-B-A. Each table entry is the average of two measurements, with no simultaneous model instances or compilation during the reported comparison.

First run: stock graph versus reusing coefficient nodes across physical-layer visits:

| Input tokens | Stock, tok/s | Shared nodes, tok/s | Change |
|---|---:|---:|---:|
| 17 | 12.98 | 16.51 | +27.3% |
| 641 | 13.08 | 13.98 | +6.9% |

Second run: shared nodes versus precomputing coefficients and placing them with layer matrices:

| Input tokens | Shared nodes, tok/s | Precomputed, tok/s | Change |
|---|---:|---:|---:|
| 17 | 15.49 | 22.46 | +45.0% |
| 641 | 15.14 | 21.50 | +42.1% |
| 2111 | 15.54 | 20.21 | +30.0% |

All generated text matched within each comparison. On a separate 16-token check, generated token IDs and returned top-5 log probabilities matched between the intermediate and precomputed builds. I also checked a local tool-call round trip: the model requested an MCP tool, received its result, and used it in the final answer.

This is a prototype with limited validation. I have not run full CI, perplexity/KL evaluation, other backends or GPUs, other loop counts, tensor splitting, or a filled 8192-token context. Per-graph coefficient allocation adds work when a graph is rebuilt, which deserves further measurement. The repeated sampling above is small and does not establish a universal throughput guarantee.

## Reproduce

Build the unmodified revision and this branch separately, using the same CUDA build options. Stop other model instances and competing GPU workloads, then run the standard-library-only harness:

```sh
python examples/dory-coefficient-benchmark.py --baseline /path/to/base/llama-server --candidate /path/to/candidate/llama-server --model /path/to/Dory-Q5_K_M.gguf --context 8192 --loops 2 --tokens 128 --output dory-results.json
```

On Windows, use `.exe` paths and optionally pass `--runtime-dir` for the CUDA runtime DLL directory. The harness checks that its port is free, launches only its own server processes, performs the warmup and identical requests, saves timings and generated text, and stops each child before starting the next one. It uses three input sizes; the historical first-stage run used only the first two. `dory-coefficient-measurements.json` records the historical per-request timings and response hashes without local filesystem paths.
