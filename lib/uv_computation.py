"""UV coverage computation engine for FITS image reconstruction.

Handles FITS file loading, FFT computation, UV point management,
and image reconstruction from sampled UV points. Separated from
the GUI layer to allow independent testing and reuse.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Distance threshold for detecting clicks on existing UV points (in UV coordinates)
CLICK_DISTANCE_THRESHOLD = 0.01

# Default limits for random UV generation
UV_RANGE_MIN = -0.5
UV_RANGE_MAX = 0.5
UV_STEP = 0.01

# Maximum number of random points per generation
MAX_RANDOM_POINTS = 3000
RANDOM_POINTS_STEP = 10


@dataclass
class UVData:
    """Container for UV point selections."""

    u: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)

    def clear(self) -> None:
        """Remove all UV points."""
        self.u.clear()
        self.v.clear()

    def add_conjugate_pair(self, u: float, v: float) -> None:
        """Add a UV point and its conjugate (u, v) and (-u, -v)."""
        self.u.append(u)
        self.v.append(v)
        self.u.append(-u)
        self.v.append(-v)

    def add_point(self, u: float, v: float) -> None:
        """Add a single UV point without conjugate."""
        self.u.append(u)
        self.v.append(v)

    def remove_point(self, index: int) -> None:
        """Remove the UV point at the given index."""
        self.u.pop(index)
        self.v.pop(index)

    @property
    def count(self) -> int:
        """Number of UV points."""
        return len(self.u)


class UVComputation:
    """Handles FITS file loading, FFT computation, and UV-based reconstruction.

    Attributes:
        image_data: 2D array of the loaded FITS image pixel data.
        freq_x: Shifted horizontal frequency axis.
        freq_y: Shifted vertical frequency axis.
        fft_data_shifted: Shifted 2D FFT of the image data.
    """

    def __init__(self) -> None:
        self.image_data: Optional[NDArray] = None
        self.freq_x: Optional[NDArray] = None
        self.freq_y: Optional[NDArray] = None
        self.fft_data_shifted: Optional[NDArray] = None
        self._prev_result: Optional[NDArray] = None

    def load_fits_file(self, path: str) -> None:
        """Load a FITS file and extract the primary image data.

        Args:
            path: Filesystem path to a FITS file.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read as FITS.
        """
        try:
            hdu_list = fits.open(path)
        except FileNotFoundError:
            logger.error("FITS file not found: %s", path)
            raise
        except OSError as exc:
            logger.error("Could not load FITS file '%s': %s", path, exc)
            raise

        try:
            self.image_data = hdu_list[0].data.squeeze()
            self._prev_result = None  # Force first render after load
        finally:
            hdu_list.close()

        logger.info("Loaded FITS file: %s (shape=%s)", path, self.image_data.shape)

    def compute_fft(self) -> None:
        """Compute the 2D FFT and frequency axes from the loaded image data.

        Must be called after load_fits_file().
        """
        if self.image_data is None:
            raise RuntimeError("No image data loaded. Call load_fits_file() first.")

        n_rows, n_cols = self.image_data.shape
        self.freq_x = np.fft.fftshift(np.fft.fftfreq(n_cols))
        self.freq_y = np.fft.fftshift(np.fft.fftfreq(n_rows))
        fft_data = np.fft.fft2(self.image_data)
        self.fft_data_shifted = np.fft.fftshift(fft_data)

    def reconstruct_image(self, uv_data: UVData) -> Optional[NDArray]:
        """Reconstruct the image from the selected UV points.

        Samples the FFT at the given UV coordinates and performs an
        inverse FFT to produce the reconstructed image.

        Args:
            uv_data: The UV point selections to sample.

        Returns:
            The reconstructed 2D image array, or None if the result
            hasn't changed since the last call.
        """
        if self.fft_data_shifted is None or self.freq_x is None or self.freq_y is None:
            raise RuntimeError("FFT not computed. Call compute_fft() first.")

        fft_sampled = np.zeros_like(self.fft_data_shifted, dtype=complex)
        for u, v in zip(uv_data.u, uv_data.v):
            u_idx = np.abs(self.freq_x - u).argmin()
            v_idx = np.abs(self.freq_y - v).argmin()
            fft_sampled[v_idx, u_idx] = self.fft_data_shifted[v_idx, u_idx]

        # Skip rendering if the sampled FFT hasn't changed
        if self._prev_result is not None and np.array_equal(self._prev_result, fft_sampled):
            return None

        self._prev_result = fft_sampled

        fft_sampled_unshifted = np.fft.ifftshift(fft_sampled)
        return np.fft.ifft2(fft_sampled_unshifted).real

    def find_nearest_point_index(
        self, uv_data: UVData, u: float, v: float
    ) -> tuple[Optional[int], float]:
        """Find the index of the nearest UV point to the given coordinates.

        Args:
            uv_data: Current UV point selections.
            u: U coordinate to search near.
            v: V coordinate to search near.

        Returns:
            A tuple of (index, distance). Index is None if uv_data is empty.
        """
        if uv_data.count == 0:
            return None, float("inf")

        distances = np.sqrt(
            (np.array(uv_data.u) - u) ** 2 + (np.array(uv_data.v) - v) ** 2
        )
        min_idx = int(np.argmin(distances))
        return min_idx, float(distances[min_idx])

    def generate_random_uv_points(
        self,
        uv_data: UVData,
        num_points: int,
        u_range: tuple[float, float],
        v_range: tuple[float, float],
    ) -> None:
        """Generate random UV points within the specified ranges.

        Args:
            uv_data: UV data container to add points to.
            num_points: Number of points to generate.
            u_range: (min, max) range for U coordinates.
            v_range: (min, max) range for V coordinates.
        """
        new_u = np.random.uniform(low=u_range[0], high=u_range[1], size=num_points)
        new_v = np.random.uniform(low=v_range[0], high=v_range[1], size=num_points)
        uv_data.u.extend(new_u.tolist())
        uv_data.v.extend(new_v.tolist())

    def reset_prev_result(self) -> None:
        """Reset the cached previous result to force re-rendering."""
        self._prev_result = None
