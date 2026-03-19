"""
InverseDesigner: generate microstructures conditioned on target material properties.

Orchestrates the full inverse design pipeline:
  1. Encode target properties as a condition vector.
  2. Run reverse diffusion (DiffusionModel.sample) to generate candidate microstructures.
  3. Evaluate generated microstructures with MaterialPropertyPredictor.
  4. Report achieved vs target properties.
"""

import numpy as np
import torch

from .diffusion import DiffusionModel
from .predictor import MaterialPropertyPredictor


class InverseDesigner:
    """
    High-level API for inverse material microstructure design.

    Given a set of target material properties, generates microstructure images
    that the model believes will exhibit those properties.

    Parameters
    ----------
    diffusion_model : DiffusionModel
        Trained (or untrained demo) DDPM model.
    predictor : MaterialPropertyPredictor or None
        Trained CNN predictor. If None, uses heuristic property estimation.
    property_names : list of str
        Names for each property dimension (for reporting).
    """

    def __init__(
        self,
        diffusion_model: DiffusionModel,
        predictor: MaterialPropertyPredictor | None = None,
        property_names: list[str] | None = None,
    ):
        self.diffusion = diffusion_model
        self.predictor = predictor
        self.property_names = property_names or [
            f"prop_{i}" for i in range(diffusion_model.n_properties)
        ]

    def design(
        self,
        target_properties: np.ndarray,
        n_samples: int = 1,
        n_steps: int | None = None,
    ) -> dict:
        """
        Generate microstructures targeting the given property values.

        Parameters
        ----------
        target_properties : np.ndarray
            Shape (n_properties,) or (B, n_properties). Values in [0, 1].
            For multiple targets, each row is an independent target.
        n_samples : int
            Number of candidate microstructures to generate per target.
        n_steps : int or None
            Reverse diffusion steps. Fewer = faster but lower quality.

        Returns
        -------
        dict with keys:
            'images'    : np.ndarray (n_targets * n_samples, 1, H, W)
            'targets'   : np.ndarray (n_targets, n_properties)
            'achieved'  : np.ndarray (n_targets * n_samples, n_properties)
            'error'     : np.ndarray (n_targets * n_samples, n_properties)
            'report'    : list of dicts (human-readable per-sample summary)
        """
        targets = np.atleast_2d(target_properties).astype(np.float32)
        n_targets = targets.shape[0]

        # Tile targets so we generate n_samples per target
        cond = np.repeat(targets, n_samples, axis=0)  # (n_targets * n_samples, n_props)

        # Generate
        images = self.diffusion.sample(cond, n_steps=n_steps)

        # Evaluate achieved properties
        achieved = self._evaluate(images)

        error = np.abs(achieved - cond)

        report = []
        for i in range(len(cond)):
            target_i = cond[i]
            achieved_i = achieved[i]
            entry = {
                "sample": i,
                "target_idx": i // n_samples,
            }
            for j, name in enumerate(self.property_names):
                entry[f"target_{name}"] = float(target_i[j])
                entry[f"achieved_{name}"] = float(achieved_i[j])
                entry[f"error_{name}"] = float(error[i, j])
            report.append(entry)

        return {
            "images": images,
            "targets": targets,
            "achieved": achieved,
            "error": error,
            "report": report,
        }

    def _evaluate(self, images: np.ndarray) -> np.ndarray:
        """Evaluate properties of generated images."""
        if self.predictor is not None:
            x = torch.from_numpy(images).float()
            self.predictor.eval()
            with torch.no_grad():
                return self.predictor(x).cpu().numpy()
        else:
            # Heuristic fallback
            props = []
            for img in images:
                h = MaterialPropertyPredictor.heuristic_properties(img)
                props.append([h["hardness"], h["conductivity"]])
            return np.array(props, dtype=np.float32)

    def print_report(self, result: dict):
        """Pretty-print a design result."""
        print(f"\n{'='*60}")
        print("Inverse Design Report")
        print(f"{'='*60}")
        for entry in result["report"]:
            print(f"\nSample {entry['sample']} (target group {entry['target_idx']}):")
            for name in self.property_names:
                t = entry[f"target_{name}"]
                a = entry[f"achieved_{name}"]
                e = entry[f"error_{name}"]
                direction = "↑" if a >= t else "↓"
                print(f"  {name:15s}: target={t:.3f}  achieved={a:.3f}  err={e:.3f} {direction}")
        print(f"\n{'='*60}")
        mean_err = np.mean(result["error"])
        print(f"Mean absolute error across all properties: {mean_err:.4f}")
        print(f"{'='*60}\n")
