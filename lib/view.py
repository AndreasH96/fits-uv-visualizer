"""GUI layer for the FITS UV coverage visualizer.

Handles all matplotlib rendering, user interaction (clicks, buttons,
sliders), and delegates computation to UVComputation.
"""

import logging
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button, RangeSlider, Slider

from lib.uv_computation import (
    CLICK_DISTANCE_THRESHOLD,
    MAX_RANDOM_POINTS,
    RANDOM_POINTS_STEP,
    UV_RANGE_MAX,
    UV_RANGE_MIN,
    UV_STEP,
    UVComputation,
    UVData,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = "./data/input_image_0.fits"
FACE_COLOR = "#F8F5E9"
BUTTON_COLOR = "#9DC08B"
BUTTON_HOVER_COLOR = "#3A7D44"


class View:
    """Main application window for the FITS UV coverage visualizer.

    Manages the matplotlib figure, subplots, widgets, and user
    interaction. Delegates all computation to a UVComputation instance.
    """

    def __init__(self, init_data_path: str = DEFAULT_DATA_PATH) -> None:
        self.uv_data = UVData()
        self.computation = UVComputation()
        self.current_data_path = init_data_path

        # Plot axes (initialized in run())
        self.uv_selection_plot: Axes
        self.fits_plot: Axes
        self.result_plot: Axes

    def load_fits_file(self, path: str | None = None) -> None:
        """Load a FITS file and prepare computation state.

        Args:
            path: Optional path override. Uses current_data_path if None.
        """
        if path:
            self.current_data_path = path

        self.computation.load_fits_file(self.current_data_path)
        self.computation.compute_fft()

    def clear_uv_selection(self, event: MouseEvent | None) -> None:
        """Clear all UV points and reset sliders."""
        self.uv_data.clear()
        self.computation.reset_prev_result()
        self.render_uv_selection_plot()
        self.u_range_slider.reset()
        self.v_range_slider.reset()
        self.num_points_slider.reset()
        self.render_result_plot()

    def generate_uv_click(self, event: MouseEvent) -> None:
        """Handle 'Generate' button click to create random UV points."""
        num_points = int(self.num_points_slider.val)
        u_range = self.u_range_slider.val
        v_range = self.v_range_slider.val

        self.computation.generate_random_uv_points(
            self.uv_data, num_points, u_range, v_range
        )
        self.render_uv_selection_plot()
        self.render_result_plot()

    def onclick(self, event: MouseEvent) -> None:
        """Handle click on the UV selection plot.

        Left-clicking near an existing point removes it. Clicking on
        empty space adds a new conjugate pair (u, v) and (-u, -v).
        """
        if not self.uv_selection_plot.in_axes(event):
            return
        if event.xdata is None or event.ydata is None:
            return

        idx, distance = self.computation.find_nearest_point_index(
            self.uv_data, event.xdata, event.ydata
        )

        if idx is not None and distance < CLICK_DISTANCE_THRESHOLD:
            self.uv_data.remove_point(idx)
        else:
            self.uv_data.add_conjugate_pair(event.xdata, event.ydata)

        self.render_uv_selection_plot()
        self.render_result_plot()

    def open_fits_selection_window(self, event: MouseEvent) -> None:
        """Open a file dialog to select and load a new FITS file."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            tkroot = tk.Tk()
            tkroot.withdraw()

            initial_dir = os.path.dirname(os.path.realpath(__file__))
            file_path = filedialog.askopenfilename(
                title="Select a FITS file",
                filetypes=[("FITS files", "*.fits")],
                initialdir=initial_dir,
            )
            tkroot.destroy()
        except ImportError:
            logger.warning("Tkinter not available; cannot open file dialog.")
            return

        if not file_path:
            return

        self.load_fits_file(file_path)
        self.clear_uv_selection(None)
        self.render_image_data()

    def render_result_plot(self) -> None:
        """Reconstruct and display the image from selected UV points."""
        reconstructed = self.computation.reconstruct_image(self.uv_data)
        if reconstructed is None:
            return  # No change from previous result

        self.result_plot.imshow(reconstructed)
        self.result_plot.set_title("Reconstruction from UV points")
        self.result_plot.figure.canvas.draw_idle()

    def render_uv_selection_plot(self) -> None:
        """Redraw the UV point selection scatter plot."""
        self.uv_selection_plot.clear()
        self.uv_selection_plot.set_title("UV selection")
        self.uv_selection_plot.set_xlabel("U")
        self.uv_selection_plot.set_ylabel("V")
        self.uv_selection_plot.scatter(
            self.uv_data.u, self.uv_data.v, c="blue", s=10
        )

        freq_x = self.computation.freq_x
        freq_y = self.computation.freq_y
        if freq_x is not None and freq_y is not None:
            self.uv_selection_plot.set_xlim(freq_x.min(), freq_x.max())
            self.uv_selection_plot.set_ylim(freq_y.min(), freq_y.max())
            self.uv_selection_plot.vlines(
                x=0, ymin=freq_y.min(), ymax=freq_y.max(),
                linestyles="--", colors="grey",
            )
            self.uv_selection_plot.hlines(
                y=0, xmin=freq_x.min(), xmax=freq_x.max(),
                linestyles="--", colors="grey",
            )

        self.uv_selection_plot.figure.canvas.draw_idle()

    def render_image_data(self) -> None:
        """Display the loaded FITS image."""
        if self.computation.image_data is None:
            return
        self.fits_plot.imshow(self.computation.image_data)
        self.fits_plot.set_title("FITS Image")
        self.fits_plot.figure.canvas.draw_idle()

    def run(self) -> None:
        """Initialize the GUI layout and start the application."""
        self.fig = plt.figure(figsize=(15, 10))
        self.main_gs = gridspec.GridSpec(8, 8)
        self.fig.set_facecolor(FACE_COLOR)

        self.uv_selection_plot = plt.subplot(self.main_gs[:5, 2:6])
        self.fits_plot = plt.subplot(self.main_gs[5:, :4])
        self.result_plot = plt.subplot(self.main_gs[5:, 4:])

        # --- Buttons ---
        buttons_gs = gridspec.GridSpecFromSubplotSpec(6, 6, self.main_gs[0:3, 0:2])

        select_fits_button = Button(
            plt.subplot(buttons_gs[0, :4]),
            "Select FITS file",
            color=BUTTON_COLOR,
            hovercolor=BUTTON_HOVER_COLOR,
        )
        select_fits_button.label.set_fontweight("semibold")
        select_fits_button.on_clicked(self.open_fits_selection_window)
        # Store reference to prevent garbage collection
        self._select_fits_button = select_fits_button

        clear_button = Button(
            plt.subplot(buttons_gs[1, :4]),
            "Clear selection",
            color=BUTTON_COLOR,
            hovercolor=BUTTON_HOVER_COLOR,
        )
        clear_button.label.set_fontweight("semibold")
        clear_button.on_clicked(self.clear_uv_selection)
        self._clear_button = clear_button

        self.uv_selection_plot.figure.canvas.mpl_connect(
            "button_press_event", self.onclick
        )

        # --- UV Generation Controls ---
        text_input_gs = gridspec.GridSpecFromSubplotSpec(
            6, 6, self.main_gs[0:3, 6:]
        )

        u_slider_ax = plt.subplot(text_input_gs[1, 1:])
        u_slider_ax.set_title("Generate UV points")
        self.u_range_slider = RangeSlider(
            u_slider_ax,
            valmin=UV_RANGE_MIN,
            valmax=UV_RANGE_MAX,
            label="U range",
            valstep=UV_STEP,
        )
        self.v_range_slider = RangeSlider(
            plt.subplot(text_input_gs[2, 1:]),
            valmin=UV_RANGE_MIN,
            valmax=UV_RANGE_MAX,
            label="V range",
            valstep=UV_STEP,
        )
        self.num_points_slider = Slider(
            plt.subplot(text_input_gs[3, 1:]),
            label="# Points",
            valmin=0,
            valmax=MAX_RANDOM_POINTS,
            valstep=RANDOM_POINTS_STEP,
        )

        generate_button = Button(
            plt.subplot(text_input_gs[4, 2:5]),
            "Generate",
            color=BUTTON_COLOR,
            hovercolor=BUTTON_HOVER_COLOR,
        )
        generate_button.label.set_fontweight("semibold")
        generate_button.on_clicked(self.generate_uv_click)
        self._generate_button = generate_button

        self.fig.subplots_adjust(wspace=0.3, hspace=0.7, top=0.95, left=0.1, bottom=0.08)

        # Load initial data and render
        self.load_fits_file()
        self.render_image_data()
        self.render_uv_selection_plot()
        self.render_result_plot()
        plt.show()
