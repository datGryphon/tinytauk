# AGENTS.md

## Project purpose

TinyTAuK is a standalone, memory-efficient AuK-Flash inference engine intended to be qualified on Bean before integration into TinyTalk.

## Hard architectural constraints

- Keep the production runtime Python/PyTorch-first until profiling proves another backend is necessary.
- Treat official AuK as a reference oracle, not a production dependency.
- Preserve compatibility with official AuK-Flash checkpoints/configuration rather than upstream Python APIs.
- Keep Qwen conditioner, AuK generator, and VAE device/dtype/quantization ownership independent.
- Do not recursively `.to()` the whole composite model.
- Avoid materializing the complete Qwen hidden-state stack if mathematically equivalent streaming layer fusion can be implemented.
- Quantize based on measured quality/runtime/memory results, not ideology. INT4 is a candidate, not a requirement.
- Do not implement GGUF/Vulkan until CPU qualification on Bean shows it is necessary.
- If CPU fails on Bean, profile first and accelerate the dominant component only.
- Do not add TinyTalk, Hermes, HTTP service, speaker-scene orchestration, or audio stitching before Bean runtime qualification.
- Do not vendor or fork large upstream frameworks when a small compatible implementation is sufficient.
- Keep model downloads and generated benchmark audio out of git.

## Development gates

1. Neptune bootstrap is reproducible.
2. Upstream AuK-Flash reference outputs are captured.
3. Unquantized TinyTAuK matches reference behavior.
4. Qwen layer-fusion memory path is improved without changing results.
5. CPU quantization profiles are benchmarked on Neptune.
6. One CPU candidate is frozen.
7. Exact candidate is benchmarked on Bean.
8. Only if Bean fails: evaluate targeted GGUF/Vulkan/other acceleration.
9. Only after Bean passes: define the TinyTalk backend contract.

## Code quality

- Prefer typed dataclasses and small protocols over framework-heavy abstractions.
- Keep imports of heavyweight ML libraries lazy where practical so config/CLI tooling stays cheap.
- Every optimization must have a benchmark or parity test demonstrating why it exists.
- Do not hide unsupported behavior behind fallbacks. Fail explicitly.
