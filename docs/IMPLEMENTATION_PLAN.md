# TinyTAuK implementation plan

## Goal

Build a standalone, memory-efficient Python/PyTorch implementation of AuK-Flash inference, optimize it first for CPU execution, qualify it on Bean, and only then integrate it into TinyTalk.

## Environment flow

1. **Neptune:** repository setup, correctness, parity, profiling, quantization experiments.
2. **Bean:** target runtime/memory qualification using the exact same repository and lock files.
3. **TinyTalk/Hermes:** integration only after Bean passes.

## Phase 0 — Bootstrap

- Local Nix flake supplies Python 3.10, uv, ffmpeg, libsndfile, compiler tooling and git-lfs.
- uv owns the Python ML dependency graph.
- Establish lint/test/benchmark scaffolding.

**Gate:** fresh clone on Neptune can enter `nix develop`, `uv sync`, and pass tests.

## Phase 1 — Upstream reference oracle

Run official AuK-Flash only as a development oracle. Capture a deterministic reference corpus and, where practical, intermediate tensors:

- Qwen conditioning output,
- AuK learned layer-fusion output,
- final diffusion latent,
- decoded waveform,
- runtime and peak RSS.

Do not make upstream `auk` a production dependency.

**Gate:** reproducible reference artifacts exist.

## Phase 2 — Unquantized TinyTAuK parity

Implement the minimum checkpoint-compatible inference stack:

```text
instruction/reference audio
        -> Qwen2.5-Omni Thinker
        -> AuK learned layer fusion
        -> Flux2Edit
        -> AuK-Flash four-step sampler
        -> BigVGANFlowVAE
        -> waveform
```

Each component independently owns device and dtype. Do not recursively move/cast the composite model.

**Gate:** TinyTAuK unquantized output is functionally equivalent to upstream AuK-Flash.

## Phase 3 — Qwen conditioning memory

Replace upstream-style retention/stacking of every Qwen hidden state with an equivalent streaming accumulation of the AuK layer-normalized learned weighted sum.

Prefer forward hooks or a minimal block wrapper before maintaining a custom Qwen implementation.

Measure conditioning error, peak RSS, and runtime.

**Gate:** equivalent conditioning with improved or neutral memory/runtime.

## Phase 4 — CPU quantization on Neptune

Benchmark components independently. Candidate progression:

1. BF16/FP32 baseline
2. INT8 weight-only
3. INT4 weight-only
4. INT8 activation + INT4 weight where supported/useful
5. mixed module policy

Start with large Linear tensors. Keep norms, small/sensitive tensors, and initially the VAE at higher precision.

Benchmark 5/10/15/20-second targets and record:

- startup/load time,
- conditioning time,
- four-step generation time,
- VAE decode time,
- total wall time and RTF,
- peak RSS,
- output quality.

Quality evaluation must include intelligibility, speaker similarity, pronunciation, prosody/emotion, artifacts and stability.

**Gate:** at least one materially smaller stable CPU profile without unacceptable quality loss.

## Phase 5 — Freeze Neptune CPU candidate

Version one candidate runtime profile and make benchmark results machine-readable JSON.

At this point Neptune should no longer be answering correctness questions; the remaining deployment question belongs to Bean.

## Phase 6 — Bean qualification

Clone the same repository on Bean and use the same lock files. Do not introduce Bean-specific code before baseline measurement.

Run the exact Neptune benchmark corpus. Measure repeated warm requests as well as first request behavior, memory growth and thermal throttling.

Normal TinyTalk speaker turns are expected to be roughly 5–15 seconds.

Target: return a normal turn in **under 8 minutes**, leaving margin inside Hermes's 10-minute HTTP timeout.

If CPU passes: stop backend optimization and proceed to TinyTalk integration.

If CPU fails: profile first.

## Phase 7 — Conditional acceleration only if Bean fails

- **Qwen dominates:** investigate existing Qwen2.5-Omni GGUF + llama.cpp/Vulkan behind a `ConditioningBackend`, including the AuK intermediate-layer fusion requirement.
- **AuK transformer dominates:** investigate an optional ggml/Vulkan generator backend, using mature diffusion/ggml implementations as references.
- **VAE dominates:** optimize the VAE independently.

Do not port the complete project to C++ preemptively.

## Phase 8 — Runtime contract freeze

Declare one supported Bean production profile and stabilize the Python API consumed by TinyTalk. Internal PyTorch/AuK implementation details must not leak through this interface.

## Phase 9 — TinyTalk integration

Add `TinyTAuKBackend` beside NeuTTS. Initially expose only:

- zero-shot TTS,
- instruct TTS,
- expressive per-speaker-turn rendering.

TinyTalk continues to own HTTP, scene/turn planning, normalization, pauses/crossfades, stitching and output encoding.

## Phase 10 — Hermes integration

Hermes should provide semantic delivery intent rather than AuK-specific prompt syntax. TinyTalk translates a stable speaker/delivery schema into TinyTAuK instructions.

## Non-negotiable decisions

- Official checkpoints are the artifact format.
- Upstream AuK is a reference oracle, not a runtime dependency.
- Python/PyTorch remains default until measurements force another backend.
- GGUF/Vulkan is an optional optimization path, not the design center.
- No Bean-specific fork.
- No TinyTalk/Hermes integration before Bean qualification.
- Every optimization requires parity/quality and runtime/memory evidence.
