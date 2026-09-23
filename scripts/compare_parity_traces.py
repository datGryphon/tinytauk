from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

ENCODER_ORDER = (
    "encoder_input_probe",
    "encoder_first_weight_g",
    "encoder_first_weight_v",
    "encoder_first_bias",
    *(f"encoder_layer_{index:02d}_probe" for index in range(21)),
)

ORDER = (
    "reference_lengths",
    "attention_mask",
    *ENCODER_ORDER,
    "reference_stats",
    "reference_latents",
    "conditioner_values",
    "initial_noise",
    "first_velocity",
    "final_latents",
    "decoded_audio",
)


def compare_float(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    a = left.detach().cpu().float().reshape(-1)
    b = right.detach().cpu().float().reshape(-1)
    diff = a - b
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    mean_abs = float(diff.abs().mean()) if diff.numel() else 0.0
    denom = float(torch.linalg.vector_norm(b))
    rel_l2 = float(torch.linalg.vector_norm(diff)) / max(denom, 1e-12)
    if a.numel() == 0:
        cosine = 1.0
    else:
        a_norm = float(torch.linalg.vector_norm(a))
        b_norm = float(torch.linalg.vector_norm(b))
        cosine = float(torch.dot(a, b)) / max(a_norm * b_norm, 1e-12)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "cosine": cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TinyTAuK and upstream AuK parity traces")
    parser.add_argument("local_trace")
    parser.add_argument("upstream_trace")
    args = parser.parse_args()

    local = torch.load(Path(args.local_trace), map_location="cpu", weights_only=True)
    upstream = torch.load(Path(args.upstream_trace), map_location="cpu", weights_only=True)

    first_material_divergence: str | None = None
    for name in ORDER:
        if name not in local or name not in upstream:
            print(f"{name:28s} MISSING")
            if first_material_divergence is None:
                first_material_divergence = name
            continue
        left = local[name]
        right = upstream[name]
        if tuple(left.shape) != tuple(right.shape):
            print(f"{name:28s} SHAPE local={tuple(left.shape)} upstream={tuple(right.shape)}")
            if first_material_divergence is None:
                first_material_divergence = name
            continue

        if not left.is_floating_point() and not right.is_floating_point():
            equal = bool(torch.equal(left, right))
            print(f"{name:28s} exact={equal}")
            if not equal and first_material_divergence is None:
                first_material_divergence = name
            continue

        metrics = compare_float(left, right)
        print(
            f"{name:28s} "
            f"max_abs={metrics['max_abs']:.6g} "
            f"mean_abs={metrics['mean_abs']:.6g} "
            f"rel_l2={metrics['rel_l2']:.6g} "
            f"cos={metrics['cosine']:.9f}"
        )
        if first_material_divergence is None and (
            not math.isfinite(metrics["rel_l2"]) or metrics["rel_l2"] > 1e-3 or metrics["cosine"] < 0.99999
        ):
            first_material_divergence = name

    print()
    if first_material_divergence is None:
        print("No material divergence found at traced boundaries.")
    else:
        print(f"First material divergence: {first_material_divergence}")


if __name__ == "__main__":
    main()
