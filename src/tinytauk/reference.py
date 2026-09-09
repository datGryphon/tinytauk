from __future__ import annotations

import hashlib
import json
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

DEFAULT_AUK_REPO = "tencent/AuK-Flash"
DEFAULT_QWEN_REPO = "Qwen/Qwen2.5-Omni-3B"
DEFAULT_INSTRUCTION = (
    "Speak in a calm, natural conversational voice: TinyTAuK reference oracle baseline generation."
)


@dataclass(frozen=True)
class ReferenceConfig:
    auk_repo: str = DEFAULT_AUK_REPO
    qwen_repo: str = DEFAULT_QWEN_REPO
    instruction: str = DEFAULT_INSTRUCTION
    gen_seconds: float = 3.0
    seed: int = 42
    output_dir: Path = Path("benchmarks/results/reference-oracle")


@dataclass(frozen=True)
class MemorySample:
    current_rss_mb: float
    peak_rss_mb: float


@dataclass(frozen=True)
class TensorArtifact:
    path: str
    shape: list[int]
    dtype: str
    numel: int
    sha256: str


@dataclass(frozen=True)
class ReferenceResult:
    auk_repo: str
    qwen_repo: str
    config_path: str
    checkpoint_path: str
    instruction: str
    requested_seconds: float
    generated_seconds: float
    sample_rate: int
    seed: int
    asset_resolve_seconds: float
    load_seconds: float
    generate_seconds: float
    realtime_factor: float
    total_seconds: float
    rss_before_load_mb: float
    rss_after_load_mb: float
    rss_after_generate_mb: float
    peak_rss_mb: float
    output_wav: str
    conditioning: TensorArtifact
    context_mask: TensorArtifact
    sampled_latent: TensorArtifact
    python: str
    platform: str
    torch: str
    auk: str
    transformers: str


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def memory_sample() -> MemorySample:
    current_kb = 0.0
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_kb = float(line.split()[1])
                break

    # Linux reports ru_maxrss in KiB.
    peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return MemorySample(current_rss_mb=current_kb / 1024.0, peak_rss_mb=peak_kb / 1024.0)


def resolve_auk_snapshot(repo_id: str) -> tuple[Path, Path]:
    snapshot = Path(snapshot_download(repo_id=repo_id))

    config_candidates = [snapshot / "config.yaml", snapshot / "config.yml"]
    config_path = next((path for path in config_candidates if path.is_file()), None)
    if config_path is None:
        yaml_files = sorted((*snapshot.glob("*.yaml"), *snapshot.glob("*.yml")))
        if len(yaml_files) != 1:
            raise RuntimeError(f"Unable to identify AuK config in {snapshot}: {yaml_files}")
        config_path = yaml_files[0]

    checkpoints = [path for path in snapshot.glob("*.safetensors") if "vae" not in path.name.lower()]
    if not checkpoints:
        raise RuntimeError(f"Unable to identify AuK checkpoint in {snapshot}")
    checkpoint_path = max(checkpoints, key=lambda path: path.stat().st_size)

    return config_path, checkpoint_path


def _messages(instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}],
        }
    ]


def _tensor_digest(tensor: torch.Tensor) -> str:
    packed = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(packed.numpy().tobytes()).hexdigest()


def _save_tensor(path: Path, tensor: torch.Tensor) -> TensorArtifact:
    value = tensor.detach().cpu().contiguous()
    torch.save(value, path)
    return TensorArtifact(
        path=str(path),
        shape=list(value.shape),
        dtype=str(value.dtype),
        numel=value.numel(),
        sha256=_tensor_digest(value),
    )


def _install_capture_hooks(engine: Any) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}

    original_encode_text = engine.model.encode_text

    def capture_encode_text(cond_inputs: Any, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mask = original_encode_text(cond_inputs, device)
        captures["conditioning"] = hidden.detach().cpu()
        captures["context_mask"] = mask.detach().cpu()
        return hidden, mask

    original_sample = engine.model.sample

    def capture_sample(*args: Any, **kwargs: Any) -> Any:
        output, trajectory = original_sample(*args, **kwargs)
        captures["sampled_latent"] = trajectory[-1].detach().cpu()
        return output, trajectory

    engine.model.encode_text = capture_encode_text
    engine.model.sample = capture_sample
    return captures


def _require_capture(captures: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        return captures[name]
    except KeyError as exc:
        raise RuntimeError(f"Reference oracle did not capture {name}") from exc


def run_reference(config: ReferenceConfig) -> ReferenceResult:
    try:
        infer_auk: Any = import_module("auk.infer.infer_auk")
    except ImportError as exc:
        raise RuntimeError(
            "Upstream AuK is not installed. Run `bash scripts/setup-reference` first."
        ) from exc

    config.output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = config.output_dir / "reference.wav"
    json_path = config.output_dir / "reference.json"

    started = time.perf_counter()
    before_load = memory_sample()

    resolve_started = time.perf_counter()
    config_path, checkpoint_path = resolve_auk_snapshot(config.auk_repo)
    asset_resolve_seconds = time.perf_counter() - resolve_started

    load_started = time.perf_counter()
    engine = infer_auk.AukInfer(
        str(config_path),
        str(checkpoint_path),
        device="cpu",
        dtype="bf16",
        qwen_path=config.qwen_repo,
    )
    load_seconds = time.perf_counter() - load_started
    after_load = memory_sample()

    captures = _install_capture_hooks(engine)

    generate_started = time.perf_counter()
    audio, sample_rate = engine.generate(
        _messages(config.instruction),
        gen_seconds=config.gen_seconds,
        seed=config.seed,
    )
    generate_seconds = time.perf_counter() - generate_started
    after_generate = memory_sample()

    infer_auk.save_audio(audio, sample_rate, str(wav_path))
    generated_seconds = float(audio.shape[-1]) / float(sample_rate)

    conditioning = _save_tensor(
        config.output_dir / "conditioning.pt",
        _require_capture(captures, "conditioning"),
    )
    context_mask = _save_tensor(
        config.output_dir / "context_mask.pt",
        _require_capture(captures, "context_mask"),
    )
    sampled_latent = _save_tensor(
        config.output_dir / "sampled_latent.pt",
        _require_capture(captures, "sampled_latent"),
    )

    result = ReferenceResult(
        auk_repo=config.auk_repo,
        qwen_repo=config.qwen_repo,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        instruction=config.instruction,
        requested_seconds=config.gen_seconds,
        generated_seconds=generated_seconds,
        sample_rate=int(sample_rate),
        seed=config.seed,
        asset_resolve_seconds=asset_resolve_seconds,
        load_seconds=load_seconds,
        generate_seconds=generate_seconds,
        realtime_factor=generate_seconds / generated_seconds,
        total_seconds=time.perf_counter() - started,
        rss_before_load_mb=before_load.current_rss_mb,
        rss_after_load_mb=after_load.current_rss_mb,
        rss_after_generate_mb=after_generate.current_rss_mb,
        peak_rss_mb=max(
            before_load.peak_rss_mb,
            after_load.peak_rss_mb,
            after_generate.peak_rss_mb,
        ),
        output_wav=str(wav_path),
        conditioning=conditioning,
        context_mask=context_mask,
        sampled_latent=sampled_latent,
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=str(torch.__version__),
        auk=_package_version("auk"),
        transformers=_package_version("transformers"),
    )
    json_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return result
