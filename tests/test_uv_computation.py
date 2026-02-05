"""Tests for the UV computation engine."""

import os

import numpy as np
import pytest

from lib.uv_computation import UVComputation, UVData

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_FITS = os.path.join(DATA_DIR, "input_image_0.fits")


# --- UVData tests ---


class TestUVData:
    def test_initial_state(self):
        data = UVData()
        assert data.count == 0
        assert data.u == []
        assert data.v == []

    def test_add_point(self):
        data = UVData()
        data.add_point(0.1, 0.2)
        assert data.count == 1
        assert data.u == [0.1]
        assert data.v == [0.2]

    def test_add_conjugate_pair(self):
        data = UVData()
        data.add_conjugate_pair(0.1, 0.2)
        assert data.count == 2
        assert data.u == [0.1, -0.1]
        assert data.v == [0.2, -0.2]

    def test_remove_point(self):
        data = UVData()
        data.add_point(0.1, 0.2)
        data.add_point(0.3, 0.4)
        data.remove_point(0)
        assert data.count == 1
        assert data.u == [0.3]
        assert data.v == [0.4]

    def test_clear(self):
        data = UVData()
        data.add_point(0.1, 0.2)
        data.add_point(0.3, 0.4)
        data.clear()
        assert data.count == 0
        assert data.u == []
        assert data.v == []


# --- UVComputation tests ---


class TestUVComputation:
    def test_load_fits_file(self):
        comp = UVComputation()
        comp.load_fits_file(SAMPLE_FITS)
        assert comp.image_data is not None
        assert comp.image_data.ndim == 2

    def test_load_fits_file_not_found(self):
        comp = UVComputation()
        with pytest.raises(FileNotFoundError):
            comp.load_fits_file("/nonexistent/path.fits")

    def test_compute_fft_without_data_raises(self):
        comp = UVComputation()
        with pytest.raises(RuntimeError, match="No image data loaded"):
            comp.compute_fft()

    def test_compute_fft(self):
        comp = UVComputation()
        comp.load_fits_file(SAMPLE_FITS)
        comp.compute_fft()
        assert comp.freq_x is not None
        assert comp.freq_y is not None
        assert comp.fft_data_shifted is not None
        # freq axes should match image dimensions
        assert len(comp.freq_x) == comp.image_data.shape[1]
        assert len(comp.freq_y) == comp.image_data.shape[0]

    def test_reconstruct_image_without_fft_raises(self):
        comp = UVComputation()
        with pytest.raises(RuntimeError, match="FFT not computed"):
            comp.reconstruct_image(UVData())

    def test_reconstruct_image_empty_uv(self):
        comp = UVComputation()
        comp.load_fits_file(SAMPLE_FITS)
        comp.compute_fft()
        result = comp.reconstruct_image(UVData())
        # First call should always return a result (even if all zeros)
        assert result is not None
        assert result.shape == comp.image_data.shape

    def test_reconstruct_image_caching(self):
        comp = UVComputation()
        comp.load_fits_file(SAMPLE_FITS)
        comp.compute_fft()
        uv = UVData()
        result1 = comp.reconstruct_image(uv)
        assert result1 is not None
        # Same UV data should return None (cached)
        result2 = comp.reconstruct_image(uv)
        assert result2 is None

    def test_reconstruct_image_with_points(self):
        comp = UVComputation()
        comp.load_fits_file(SAMPLE_FITS)
        comp.compute_fft()
        uv = UVData()
        uv.add_conjugate_pair(0.0, 0.0)
        result = comp.reconstruct_image(uv)
        assert result is not None
        assert result.shape == comp.image_data.shape
        # With at least the DC component, reconstruction shouldn't be all zeros
        assert not np.allclose(result, 0)

    def test_find_nearest_point_empty(self):
        comp = UVComputation()
        idx, dist = comp.find_nearest_point_index(UVData(), 0.0, 0.0)
        assert idx is None
        assert dist == float("inf")

    def test_find_nearest_point(self):
        comp = UVComputation()
        uv = UVData()
        uv.add_point(0.1, 0.2)
        uv.add_point(0.5, 0.5)
        idx, dist = comp.find_nearest_point_index(uv, 0.11, 0.21)
        assert idx == 0
        assert dist < 0.02

    def test_generate_random_uv_points(self):
        comp = UVComputation()
        uv = UVData()
        comp.generate_random_uv_points(uv, 100, (-0.3, 0.3), (-0.2, 0.2))
        assert uv.count == 100
        assert all(-0.3 <= u <= 0.3 for u in uv.u)
        assert all(-0.2 <= v <= 0.2 for v in uv.v)

    def test_generate_random_extends_existing(self):
        comp = UVComputation()
        uv = UVData()
        uv.add_point(0.0, 0.0)
        comp.generate_random_uv_points(uv, 50, (-0.5, 0.5), (-0.5, 0.5))
        assert uv.count == 51

    def test_reset_prev_result(self):
        comp = UVComputation()
        comp.load_fits_file(SAMPLE_FITS)
        comp.compute_fft()
        uv = UVData()
        comp.reconstruct_image(uv)
        # After reset, same UV data should produce a result again
        comp.reset_prev_result()
        result = comp.reconstruct_image(uv)
        assert result is not None
