# Speech quality benchmark

TinyTAuK uses a fixed corpus to compare optimization candidates.

Each case in `benchmarks/quality/cases.json` defines:

- target text;
- full AuK instruction;
- generation duration;
- deterministic seed.

## Render

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
bash scripts/quality-corpus
```

Default policies are `fp32`, `int8-attn`, `int8-core`, and `int8-all`. Set
`MODES` to restrict a run.

The renderer writes WAVs plus timing/RMS metadata under
`benchmarks/results/quality-corpus/`. Conditioning fixtures are keyed to the
full case definition and rebuilt when a case changes.

## Score

```bash
bash scripts/score-quality-corpus benchmarks/results/quality-corpus
```

The scorer reports:

- WER/CER against the target text;
- WER/CER against the frozen FP32 transcript;
- candidate WER/CER deltas from FP32.

WER/CER use Levenshtein edit distance after text normalization. Corpus scores
sum edits and reference units across all utterances.

The default ASR model is `openai/whisper-small.en`. Override it with
`QUALITY_ASR_MODEL`; use `--retranscribe` to rebuild cached transcripts.

For a stronger CUDA evaluator:

```bash
QUALITY_ASR_MODEL=openai/whisper-large-v3-turbo \
QUALITY_DEVICE=cuda \
bash scripts/score-quality-corpus benchmarks/results/quality-corpus --retranscribe
```

The scorer level-normalizes audio before ASR by default so loudness changes do
not dominate intelligibility scoring. Raw RMS is still recorded by the
renderer.

## Acceptance

WER/CER alone are not enough. Keep a candidate only when it combines useful
performance or memory savings with acceptable intelligibility, listening
quality, and acoustic regression metrics.
