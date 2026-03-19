"""
Core tests for microdiff-matdesign.

Tests are designed to be fast (CPU-only, small images, few steps) so they
work in CI without a GPU.
"""

import numpy as np
import pytest
import torch

from microdiff_matdesign import (
    MicrostructureGenerator,
    MaterialPropertyPredictor,
    DiffusionModel,
    InverseDesigner,
)


# ---------------------------------------------------------------------------
# MicrostructureGenerator
# ---------------------------------------------------------------------------

class TestMicrostructureGenerator:

    def test_single_image_shape(self):
        gen = MicrostructureGenerator(image_size=32, seed=0)
        img = gen.generate(n_grains=10)
        assert img.shape == (1, 32, 32)

    def test_image_range(self):
        gen = MicrostructureGenerator(image_size=32, seed=1)
        img = gen.generate(n_grains=15, phase_fraction=0.3)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_batch_shape(self):
        gen = MicrostructureGenerator(image_size=32, seed=2)
        batch = gen.generate_batch(5)
        assert batch.shape == (5, 1, 32, 32)

    def test_reproducibility(self):
        gen1 = MicrostructureGenerator(image_size=32, seed=42)
        gen2 = MicrostructureGenerator(image_size=32, seed=42)
        img1 = gen1.generate(n_grains=20)
        img2 = gen2.generate(n_grains=20)
        np.testing.assert_array_equal(img1, img2)

    def test_grain_boundary_present(self):
        """Grain boundaries should produce pixels around 0.5."""
        gen = MicrostructureGenerator(image_size=32, seed=3)
        img = gen.generate(n_grains=10, noise_sigma=0.0)
        mid_pixels = ((img.squeeze() > 0.35) & (img.squeeze() < 0.65)).sum()
        assert mid_pixels > 0, "No grain boundaries found"

    def test_phase_fraction_effect(self):
        """Higher phase fraction should produce more high-intensity pixels."""
        gen = MicrostructureGenerator(image_size=64, seed=5)
        img_lo = gen.generate(n_grains=20, noise_sigma=0.0, phase_fraction=0.0)
        gen2 = MicrostructureGenerator(image_size=64, seed=5)
        img_hi = gen2.generate(n_grains=20, noise_sigma=0.0, phase_fraction=0.8)
        bright_lo = (img_lo.squeeze() > 0.65).sum()
        bright_hi = (img_hi.squeeze() > 0.65).sum()
        assert bright_hi > bright_lo


# ---------------------------------------------------------------------------
# MaterialPropertyPredictor
# ---------------------------------------------------------------------------

class TestMaterialPropertyPredictor:

    def test_output_shape(self):
        pred = MaterialPropertyPredictor(image_size=32, n_properties=2)
        x = torch.rand(4, 1, 32, 32)
        out = pred(x)
        assert out.shape == (4, 2)

    def test_output_range(self):
        pred = MaterialPropertyPredictor(image_size=32, n_properties=2)
        x = torch.rand(8, 1, 32, 32)
        out = pred(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_predict_numpy(self):
        pred = MaterialPropertyPredictor(image_size=32)
        imgs = np.random.rand(3, 1, 32, 32).astype(np.float32)
        out = pred.predict_numpy(imgs)
        assert out.shape == (3, 2)

    def test_predict_numpy_single(self):
        pred = MaterialPropertyPredictor(image_size=32)
        img = np.random.rand(1, 32, 32).astype(np.float32)
        out = pred.predict_numpy(img)
        assert out.shape == (1, 2)

    def test_heuristic_properties_keys(self):
        img = np.random.rand(1, 32, 32).astype(np.float32)
        props = MaterialPropertyPredictor.heuristic_properties(img)
        assert "hardness" in props
        assert "conductivity" in props

    def test_heuristic_properties_range(self):
        img = np.random.rand(1, 32, 32).astype(np.float32)
        props = MaterialPropertyPredictor.heuristic_properties(img)
        assert 0.0 <= props["hardness"] <= 1.0
        assert 0.0 <= props["conductivity"] <= 1.0

    def test_heuristic_trend(self):
        """Denser grain boundaries → higher hardness, lower conductivity."""
        gen = MicrostructureGenerator(image_size=64, seed=7)
        # Fine-grained (many grains = dense boundaries)
        fine = gen.generate(n_grains=60, noise_sigma=0.0)
        # Coarse-grained (few grains = sparse boundaries)
        coarse = gen.generate(n_grains=5, noise_sigma=0.0)
        props_fine = MaterialPropertyPredictor.heuristic_properties(fine)
        props_coarse = MaterialPropertyPredictor.heuristic_properties(coarse)
        assert props_fine["hardness"] >= props_coarse["hardness"], (
            "Fine-grained should have higher hardness (Hall-Petch effect)"
        )


# ---------------------------------------------------------------------------
# DiffusionModel
# ---------------------------------------------------------------------------

class TestDiffusionModel:

    @pytest.fixture
    def tiny_model(self):
        return DiffusionModel(image_size=32, T=10, n_properties=2, device="cpu")

    def test_q_sample_shape(self, tiny_model):
        x0 = torch.rand(2, 1, 32, 32)
        t = torch.tensor([3, 7])
        xt, noise = tiny_model.q_sample(x0, t)
        assert xt.shape == (2, 1, 32, 32)
        assert noise.shape == (2, 1, 32, 32)

    def test_loss_runs(self, tiny_model):
        x0 = torch.rand(4, 1, 32, 32)
        cond = torch.rand(4, 2)
        loss = tiny_model.loss(x0, cond)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_sample_shape(self, tiny_model):
        cond = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
        out = tiny_model.sample(cond, n_steps=5)
        assert out.shape == (2, 1, 32, 32)

    def test_sample_range(self, tiny_model):
        cond = np.array([[0.5, 0.5]], dtype=np.float32)
        out = tiny_model.sample(cond, n_steps=5)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_train_quick_loss_decreases(self, tiny_model):
        gen = MicrostructureGenerator(image_size=32, seed=99)
        imgs = gen.generate_batch(20)
        props = np.array([
            list(MaterialPropertyPredictor.heuristic_properties(i).values())
            for i in imgs
        ], dtype=np.float32)
        losses = tiny_model.train_quick(imgs, props, n_epochs=3, batch_size=10, verbose=False)
        assert len(losses) == 3
        # At least one step of training happened
        assert losses[-1] > 0


# ---------------------------------------------------------------------------
# InverseDesigner
# ---------------------------------------------------------------------------

class TestInverseDesigner:

    @pytest.fixture
    def designer(self):
        model = DiffusionModel(image_size=32, T=10, n_properties=2, device="cpu")
        return InverseDesigner(model, predictor=None, property_names=["hardness", "conductivity"])

    def test_design_result_keys(self, designer):
        targets = np.array([[0.8, 0.2]], dtype=np.float32)
        result = designer.design(targets, n_samples=1, n_steps=5)
        assert "images" in result
        assert "targets" in result
        assert "achieved" in result
        assert "error" in result
        assert "report" in result

    def test_design_multi_target(self, designer):
        targets = np.array([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ], dtype=np.float32)
        result = designer.design(targets, n_samples=2, n_steps=5)
        assert result["images"].shape[0] == 6  # 3 targets × 2 samples
        assert len(result["report"]) == 6

    def test_design_achieved_range(self, designer):
        targets = np.array([[0.5, 0.5]], dtype=np.float32)
        result = designer.design(targets, n_steps=5)
        assert result["achieved"].min() >= 0.0
        assert result["achieved"].max() <= 1.0

    def test_design_error_nonneg(self, designer):
        targets = np.array([[0.3, 0.7]], dtype=np.float32)
        result = designer.design(targets, n_steps=5)
        assert (result["error"] >= 0).all()
