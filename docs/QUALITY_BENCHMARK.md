# TinyTAuK speech-quality benchmark

TinyTAuK uses a fixed speech corpus to evaluate optimization candidates independently from raw tensor similarity.

## What is measured

Every corpus case stores a canonical `spoken_text`, the complete AuK instruction, a deterministic seed, and an explicit generation duration. Bean renders the same cases for each generator policy and retains the resulting WAV files.

The offline scorer then records two distinct kinds of speech quality:

- **Absolute intelligibility:** WER and CER between the canonical spoken text and the ASR transcript.
- **FP32 regression consistency:** WER and CER between the frozen FP32 ASR transcript and each candidate ASR transcript, plus the candidate's WER/CER delta against the canonical FP32 score.

This prevents the FP32 model's own pronunciation mistakes from becoming ground truth while still measuring whether an optimization changes what the baseline says.

All WER/CER values use proper Levenshtein edit distance. Corpus scores sum edit errors and reference units across utterances rather than averaging per-utterance percentages.

## Render on Bean

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
bash scripts/bean-quality-corpus
```

Default policies are `fp32`, `int8-attn`, `int8-core`, and `int8-all`. Set `MODES` to restrict a run. Conditioning tensors under `benchmarks/results/quality-corpus/_conditioning/` are benchmark fixtures tied to each utterance; they are not reusable production speaker caches.

The renderer writes each policy's WAV files and `render.json`, plus a top-level `render-summary.json`. ASR is deliberately excluded so it cannot contaminate Bean timing measurements.

## Score off Bean

Copy `benchmarks/results/quality-corpus/` to a machine suitable for ASR and run:

```bash
bash scripts/score-quality-corpus benchmarks/results/quality-corpus
```

The default recognizer is `openai/whisper-small.en` through Transformers. `QUALITY_DEVICE=auto` selects CUDA when available, otherwise CPU. Override the model with `QUALITY_ASR_MODEL` if needed.

The scorer writes:

- `transcripts.json` — cached ASR output, including the frozen FP32 transcripts.
- `quality-scores.json` — per-case and aggregate WER/CER results.

Use `--retranscribe` to intentionally regenerate all ASR transcripts. Changing the ASR model also invalidates the transcript cache automatically.

## Interpreting candidates

Tensor and waveform SNR/cosine remain useful diagnostics, but they are not acceptance criteria. A useful quantization candidate should combine a meaningful Bean speedup with low canonical WER/CER, small WER/CER regression against FP32, and manual listening on retained WAV artifacts.
