#!/usr/bin/env python3
"""Audit whether the current CSS proxy mostly measures sequence length/energy."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def correlation(left, right):
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rms_score(features):
    flattened = features.reshape(len(features), -1)
    return np.linalg.norm(flattened, axis=1) / np.sqrt(flattened.shape[1])


def active_steps(features):
    reduce_axes = tuple(range(2, features.ndim))
    return (np.abs(features).sum(axis=reduce_axes) > 1e-8).sum(axis=1)


def add_gaussian_noise(features, severity, rng):
    if severity == 0:
        return features.copy()
    active = np.abs(features).sum(axis=-1, keepdims=True) > 1e-8
    nonzero = features[active.repeat(features.shape[-1], axis=-1)]
    scale = float(nonzero.std()) if nonzero.size else float(features.std())
    noise = rng.normal(0.0, severity * scale, size=features.shape).astype(np.float32)
    return features + noise * active


def audit_noise_response(vision, audio):
    severities = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    mean_scores = []
    sample_scores = []
    for severity in severities:
        rng = np.random.default_rng(20260720)
        noisy_vision = add_gaussian_noise(vision, float(severity), rng)
        noisy_audio = add_gaussian_noise(audio, float(severity), rng)
        scores = (rms_score(noisy_vision) + rms_score(noisy_audio)) / 2.0
        mean_scores.append(float(scores.mean()))
        sample_scores.append(scores)

    clean_scores = sample_scores[0]
    high_noise_scores = sample_scores[-1]
    increase_rate = float(np.mean(high_noise_scores > clean_scores))
    severity_correlation = correlation(severities, mean_scores)
    formatted = ", ".join(
        f"{severity:.2f}:{score:.6f}"
        for severity, score in zip(severities, mean_scores)
    )
    print(f"gaussian severity:mean_css={formatted}")
    print(f"corr(severity, mean_css)={severity_correlation:.6f}")
    print(f"fraction(css_high_noise > css_clean)={increase_rate:.6f}")


def audit_split(data, split):
    vision = np.nan_to_num(np.asarray(data["vision"], dtype=np.float32))
    audio = np.nan_to_num(np.asarray(data["audio"], dtype=np.float32))
    labels = np.asarray(data["regression_labels"], dtype=np.float32).reshape(-1)

    vision_score = rms_score(vision)
    audio_score = rms_score(audio)
    css_score = (vision_score + audio_score) / 2.0
    mean_active = (active_steps(vision) + active_steps(audio)) / 2.0

    print(f"[{split}] n={len(labels)}")
    print(f"corr(css, active_steps)={correlation(css_score, mean_active):.6f}")
    print(f"corr(css, abs_label)={correlation(css_score, np.abs(labels)):.6f}")
    print(f"corr(css, label)={correlation(css_score, labels):.6f}")
    print(f"corr(vision_score, audio_score)={correlation(vision_score, audio_score):.6f}")
    print(
        "css mean/std/min/max="
        f"{css_score.mean():.6f}/{css_score.std():.6f}/"
        f"{css_score.min():.6f}/{css_score.max():.6f}"
    )
    audit_noise_response(vision, audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        default="MFON/MOSI/aligned_50.pkl",
        help="Path to the aligned dataset pickle.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)

    for split in ("train", "valid", "test"):
        if split in dataset:
            audit_split(dataset[split], split)


if __name__ == "__main__":
    main()
