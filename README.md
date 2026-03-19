# microdiff-matdesign

**Diffusion models for inverse material microstructure design.**

Given target material properties (hardness, conductivity), generate the microstructure that should achieve them — using a conditioned denoising diffusion probabilistic model (DDPM).

---

## The Problem

Materials science has a *forward* problem that's well-studied: given a microstructure, predict its properties. The *inverse* problem is harder: given desired properties, find a microstructure that achieves them.

Diffusion models are natural here. Instead of searching the microstructure space manually, we train a generative model conditioned on target properties and sample from it.

---

## What This Does

```
target properties  →  [InverseDesigner]  →  microstructure image
   (hardness=0.8)     [DiffusionModel]      (2D grain boundary map)
   (conductivity=0.2) [DDPM reverse]
```

**Four core components:**

| Class | Role |
|-------|------|
| `MicrostructureGenerator` | Synthetic 2D microstructure images via Voronoi tessellation |
| `MaterialPropertyPredictor` | Small CNN: image → property vector |
| `DiffusionModel` | DDPM with property conditioning + classifier-free guidance |
| `InverseDesigner` | High-level API: target properties → microstructure images |

---

## Physics Background

### Microstructure generation
Polycrystalline materials consist of grains separated by grain boundaries. We approximate these using **Voronoi tessellation** of random seed points — each cell is a grain, boundaries are cell edges.

Two phases are modeled:
- **Primary phase** (gray) — base material
- **Secondary phase** (white) — precipitates, inclusions, or a second alloy phase
- **Grain boundaries** — mid-gray pixels at cell edges

### Property proxies
- **Hardness** ∝ grain boundary density + secondary phase fraction (Hall-Petch strengthening)
- **Conductivity** ∝ 1 − (grain boundary density + secondary phase fraction)

Fine-grained microstructures (many small grains) → high boundary density → harder, less conductive.

---

## Architecture

### Diffusion Model (DDPM)
- **Forward process**: `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)`
- **Reverse process**: learned noise predictor `ε_θ(x_t, t, c)` where `c` is the property condition
- **Schedule**: cosine β schedule (Nichol & Dhariwal 2021)
- **Guidance**: classifier-free guidance (Ho & Salimans 2022) — scale controls condition strength

### U-Net Noise Predictor
```
(1, 64, 64) → [enc: 32→64→128] → [bottleneck] → [dec: 128→64→32] → (1, 64, 64)
```
- 3 downsampling stages (stride-2 conv), 3 upsampling (transpose conv + skip connections)
- Time step embedded via sinusoidal encoding → MLP → injected at every ResBlock
- Property condition projected → MLP → injected at every ResBlock

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy scipy

# Run the demo (generates dataset, trains, runs inverse design)
python demo.py
```

The demo:
1. Generates 200 synthetic microstructures with random grain counts and phase fractions
2. Computes heuristic property labels (hardness, conductivity)
3. Trains the DDPM for 10 epochs
4. Generates 5 microstructures targeting hardness values from 0.1 to 0.9
5. Evaluates and reports achieved vs target properties

> **Note:** 10 epochs is a smoke-test. Real convergence requires ~100–500 epochs on a larger dataset. The model clearly learns the conditioning direction early but needs more training to achieve quantitative accuracy.

---

## Usage

```python
from microdiff_matdesign import (
    MicrostructureGenerator,
    MaterialPropertyPredictor,
    DiffusionModel,
    InverseDesigner,
)
import numpy as np

# Generate training data
gen = MicrostructureGenerator(image_size=64, seed=42)
images = gen.generate_batch(1000, n_grains_range=(8, 60))

# Compute properties
props = np.array([
    list(MaterialPropertyPredictor.heuristic_properties(img).values())
    for img in images
], dtype="float32")  # shape (1000, 2)

# Train diffusion model
model = DiffusionModel(image_size=64, T=500, device="cuda")
model.train_quick(images, props, n_epochs=100, batch_size=32)

# Inverse design: generate hard, conductive microstructure
designer = InverseDesigner(model, property_names=["hardness", "conductivity"])
result = designer.design(
    target_properties=np.array([0.8, 0.6]),  # hard + somewhat conductive
    n_samples=4,
)
designer.print_report(result)
# result["images"]: shape (4, 1, 64, 64)
```

---

## Project Structure

```
microdiff_matdesign/
  __init__.py          — public API
  generator.py         — MicrostructureGenerator (Voronoi)
  predictor.py         — MaterialPropertyPredictor (CNN)
  diffusion.py         — DiffusionModel (DDPM + U-Net)
  designer.py          — InverseDesigner (pipeline orchestration)
tests/
  test_core.py         — 22 tests covering all components
demo.py                — end-to-end demonstration
```

---

## References

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021) — [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- Ho & Salimans (2022) — [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- Hall (1951), Petch (1953) — Hall-Petch relationship (grain size → yield strength)

---

## License

Apache 2.0
