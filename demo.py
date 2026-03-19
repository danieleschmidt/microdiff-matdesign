#!/usr/bin/env python3
"""
demo.py — MicroDiff MatDesign inverse design demonstration.

Workflow:
  1. Generate synthetic microstructure dataset using Voronoi tessellation.
  2. Compute heuristic property labels (hardness, conductivity proxies).
  3. Quick-train the DDPM diffusion model on this dataset.
  4. Use InverseDesigner to generate 5 microstructures targeting different
     hardness values. Verify that the property predictor confirms correct trends.

Run with:
    ~/anaconda3/bin/python3 demo.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from microdiff_matdesign import (
    MicrostructureGenerator,
    MaterialPropertyPredictor,
    DiffusionModel,
    InverseDesigner,
)


def main():
    print("=" * 60)
    print("MicroDiff MatDesign — Inverse Design Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate synthetic microstructure dataset
    # ------------------------------------------------------------------
    print("\n[1/5] Generating synthetic microstructure dataset...")
    gen = MicrostructureGenerator(image_size=64, seed=42)
    N = 200
    images = gen.generate_batch(N, n_grains_range=(8, 60), phase_fraction_range=(0.0, 0.5))
    print(f"  Generated {N} microstructures: shape={images.shape}")

    # ------------------------------------------------------------------
    # 2. Compute heuristic property labels
    # ------------------------------------------------------------------
    print("\n[2/5] Computing heuristic property labels...")
    props = np.array([
        [v for v in MaterialPropertyPredictor.heuristic_properties(img).values()]
        for img in images
    ], dtype=np.float32)
    print(f"  Properties shape: {props.shape}")
    print(f"  Hardness range:     [{props[:, 0].min():.3f}, {props[:, 0].max():.3f}]")
    print(f"  Conductivity range: [{props[:, 1].min():.3f}, {props[:, 1].max():.3f}]")

    # ------------------------------------------------------------------
    # 3. Quick-train the diffusion model
    # ------------------------------------------------------------------
    print("\n[3/5] Training diffusion model (quick, 10 epochs)...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = DiffusionModel(
        image_size=64,
        T=200,
        n_properties=2,
        device=device,
        guidance_scale=3.0,
    )
    losses = model.train_quick(
        dataset=images,
        conditions=props,
        n_epochs=10,
        lr=2e-3,
        batch_size=32,
        verbose=True,
    )
    print(f"  Final loss: {losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Inverse design: generate microstructures for 5 target hardness values
    # ------------------------------------------------------------------
    print("\n[4/5] Running inverse design for 5 target hardness values...")
    targets = np.array([
        [0.1, 0.9],  # soft, high conductivity
        [0.3, 0.7],  # medium-soft
        [0.5, 0.5],  # balanced
        [0.7, 0.3],  # medium-hard
        [0.9, 0.1],  # hard, low conductivity
    ], dtype=np.float32)

    designer = InverseDesigner(
        diffusion_model=model,
        predictor=None,  # use heuristic evaluation for demo
        property_names=["hardness", "conductivity"],
    )

    result = designer.design(targets, n_samples=1, n_steps=100)
    designer.print_report(result)

    # ------------------------------------------------------------------
    # 5. Verify trend: hardness should increase monotonically
    # ------------------------------------------------------------------
    print("\n[5/5] Verifying hardness trend across designs...")
    achieved_hardness = result["achieved"][:, 0]
    target_hardness = targets[:, 0]

    print(f"  {'Target':>8}  {'Achieved':>10}  {'Direction':>12}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}")
    for i, (t, a) in enumerate(zip(target_hardness, achieved_hardness)):
        ok = "✓" if i == 0 or a >= achieved_hardness[i - 1] - 0.05 else "✗"
        print(f"  {t:8.3f}  {a:10.3f}  {ok}")

    # Rank correlation check
    from scipy.stats import spearmanr
    rho, pval = spearmanr(target_hardness, achieved_hardness)
    print(f"\n  Spearman ρ (target vs achieved hardness): {rho:.3f}  (p={pval:.3f})")

    if rho > 0:
        print("  ✓ Positive correlation — diffusion model learned conditioning direction")
    else:
        print("  ⚠ Weak/negative correlation — model needs more training epochs")
        print("    (Expected for a 10-epoch demo; increase epochs for real use)")

    print("\nDemo complete. See README.md for full usage details.")
    return result


if __name__ == "__main__":
    main()
