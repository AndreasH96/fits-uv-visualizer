import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button,RangeSlider, Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import numpy as np
from astropy.io import fits
import tkinter as tk
from tkinter import filedialog
import os 

default_path = './data/input_image_0.fits'
tkroot = tk.Tk()
tkroot.withdraw()
face_color = '#F8F5E9'
button_color = '#9DC08B'
button_hover_color = '#3A7D44'
track_color = '#DF6D14'

class View():
    def __init__(self,init_data_path:str=default_path):
        
        self.current_uv_data = {'u':[],'v':[]}
        self.current_data_path = init_data_path
        self.prev_result = np.zeros(shape=(1))
        self.image_data = None

        
    def load_fits_file(self,path=None):
        if path:
            self.current_data_path = path
        try:
            hdu_list = fits.open(self.current_data_path)
        except:
            print("Could not load FITS file: {path}")
        # Access the primary HDU
        self.image_data = hdu_list[0].data.squeeze()
        self.prev_result = np.zeros_like(self.image_data)
        # Close the FITS file
        hdu_list.close()

    def update_data_state(self):
        N, M = self.image_data.shape 

        # Compute frequency axes
        self.freq_x = np.fft.fftshift(np.fft.fftfreq(M))  # Horizontal U
        self.freq_y = np.fft.fftshift(np.fft.fftfreq(N))  # Vertical V
        self.fft_data = np.fft.fft2(self.image_data)
        self.fft_data_shifted = np.fft.fftshift(self.fft_data)  

    def clear_uv_selection(self,event):

        self.current_uv_data = {'u':[],'v':[]}
        self.render_uv_selection_plot()
        self.u_range_slider.reset()
        self.v_range_slider.reset()
        self.num_points_slider.reset()
        self.render_result_plot()

     
    def generate_uv_click(self,event:MouseEvent):
        num_points = int(self.num_points_slider.val)
        
        u_min, u_max = self.u_range_slider.val
        v_min, v_max = self.v_range_slider.val
        new_u = np.random.uniform(low=u_min,high=u_max,size=num_points)
        new_v = np.random.uniform(low=v_min,high=v_max,size=num_points)
        self.current_uv_data['u'].extend(new_u)
        self.current_uv_data['v'].extend(new_v)
        self.render_uv_selection_plot()
        self.render_result_plot()

    def onclick(self,event:MouseEvent):
        if self.uv_selection_plot.in_axes(event):
            if event.xdata != None and event.ydata != None:
                min_distance = 1
                if len(self.current_uv_data['u']) > 0:
                    distances = np.sqrt((self.current_uv_data['u'] - event.xdata)**2 + (self.current_uv_data['v'] - event.ydata)**2)
                    min_idx = np.argmin(distances)  # Get index of closest point
                    min_distance = distances[min_idx]

                if min_distance < 0.01:
                    self.current_uv_data['u'].pop(min_idx)
                    self.current_uv_data['v'].pop(min_idx) 

                else:
                    self.current_uv_data['u'].append(event.xdata)
                    self.current_uv_data['v'].append(event.ydata)
                    self.current_uv_data['u'].append(event.xdata * -1)
                    self.current_uv_data['v'].append(event.ydata * -1)
                    
                self.render_uv_selection_plot()
                self.render_result_plot()

    def open_fits_selection_window(self,event):
        
        initial_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = filedialog.askopenfilename(title="Select a FITS file",
                                            filetypes=[("FITS files", "*.fits")],initialdir=initial_dir)
        self.load_fits_file(file_path)
        self.update_data_state()
        self.clear_uv_selection(None)
        self.render_image_data()



    def render_result_plot(self):
        # Convert to index positions in the frequency matrix
        fft_sampled = np.zeros_like(self.fft_data_shifted, dtype=complex)
        for u, v in zip(self.current_uv_data['u'],self.current_uv_data['v']):
            u_idx = (np.abs(self.freq_x - u)).argmin()  # Find closest U index
            v_idx = (np.abs(self.freq_y - v)).argmin()  # Find closest V index
            fft_sampled[v_idx, u_idx] = self.fft_data_shifted[v_idx, u_idx]
        
        # If no change to sample (probably a lot of sampled points), don't render
        if (self.prev_result==fft_sampled).all():
            pass
        # Plot the reconstruction
        fft_sampled_unshifted = np.fft.ifftshift(fft_sampled)  # Undo shift
        reconstructed_image = np.fft.ifft2(fft_sampled_unshifted).real  # Inverse transform
        self.result_plot.imshow(reconstructed_image)
        self.result_plot.set_title("Reconstruction from UV points")
        self.result_plot.figure.canvas.draw_idle()
        self.prev_result = fft_sampled



    def render_uv_selection_plot(self):
        self.uv_selection_plot.clear()
        self.uv_selection_plot.set_title('UV selection')
        self.uv_selection_plot.set_xlabel('U')
        self.uv_selection_plot.set_ylabel('V')
        self.uv_selection_plot.scatter(self.current_uv_data['u'],self.current_uv_data['v'],c='blue')
        self.uv_selection_plot.set_xlim(self.freq_x.min(),self.freq_x.max())
        self.uv_selection_plot.set_ylim(self.freq_y.min(),self.freq_y.max())
        self.uv_selection_plot.vlines(x=0,ymin=self.freq_y.min(),ymax=self.freq_y.max(),linestyles='--',colors='grey')
        self.uv_selection_plot.hlines(y=0,xmin=self.freq_x.min(),xmax=self.freq_x.max(),linestyles='--',colors='grey')
        self.uv_selection_plot.figure.canvas.draw_idle()
        
        

    def render_image_data(self):

        self.fits_plot.imshow(self.image_data)
        self.fits_plot.set_title('FITS Image')
        self.fits_plot.figure.canvas.draw_idle()

    def run(self):

        self.fig = plt.figure(figsize=(15,10))
        #self.fig.canvas.manager.set_window_title('FITS UV selection reconstruction')
        self.main_gs = gridspec.GridSpec(8, 8)
        self.fig.set_facecolor('#F8F5E9')
        self.uv_selection_plot: Axes= plt.subplot(self.main_gs[:5, 2:6],) 
        self.fits_plot: Axes= plt.subplot(self.main_gs[5:,:4 ]) 
        self.result_plot: Axes= plt.subplot(self.main_gs[5:, 4:]) 

        self.buttons_gs = gridspec.GridSpecFromSubplotSpec(6,6,self.main_gs[0:3, 0:2])

        select_fits_button = Button(plt.subplot(self.buttons_gs[0, :4]), 'Select FITS file',color=button_color,hovercolor=button_hover_color)
        select_fits_button.label.set_fontweight('semibold')
        select_fits_button.on_clicked(self.open_fits_selection_window)


        self.clear_uv_button = Button(plt.subplot(self.buttons_gs[1, :4]), 'Clear selection',color=button_color,hovercolor=button_hover_color)
        self.clear_uv_button.label.set_fontweight('semibold')
        self.clear_uv_button.on_clicked(self.clear_uv_selection)

        self.uv_cid = self.uv_selection_plot.figure.canvas.mpl_connect('button_press_event', self.onclick)

        # Setup UV generation input area
        self.text_input_gs = gridspec.GridSpecFromSubplotSpec(6,6,self.main_gs[0:3, 6:])

        self.u_range_slider_ax = plt.subplot(self.text_input_gs[1,1:])
        self.u_range_slider_ax.set_title('Generate UV points')
        self.u_range_slider = RangeSlider(self.u_range_slider_ax,valmin=-.5,valmax=.5,label='U range',valstep=0.01)
        self.v_range_slider = RangeSlider(plt.subplot(self.text_input_gs[2,1:]),valmin=-.5,valmax=.5,label='V range',valstep=0.01)
        self.num_points_slider = Slider(plt.subplot(self.text_input_gs[3,1:]),label='# Points',valmin=0,valmax=3000,valstep=10)

        self.generate_uv_points_button = Button(plt.subplot(self.text_input_gs[4,2:5]), 'Generate',color=button_color,hovercolor=button_hover_color)
        self.generate_uv_points_button.label.set_fontweight('semibold')
        self.generate_uv_points_button.on_clicked(self.generate_uv_click)

        self.fig.subplots_adjust(wspace=0.3,hspace=.7,top=.95,left=.1,bottom=.08)  

        self.load_fits_file()
        self.update_data_state()
        self.render_image_data()
        self.render_uv_selection_plot()
        self.render_result_plot()
        plt.show()