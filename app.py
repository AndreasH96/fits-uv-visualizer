
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
import numpy as np
from astropy.io import fits
import tkinter as tk
from tkinter import filedialog
import os 
tkroot = tk.Tk()
tkroot.withdraw()

hover_over_uv_selection = False

selected_uv_points = {'x':[],'y':[]}

default_path = './data/input_image_0.fits'


def update_data_state():
    global image_data,N,M,freq_x,freq_y,fft_data,fft_data_shifted
    N, M = image_data.shape 

    # Compute frequency axes
    freq_x = np.fft.fftshift(np.fft.fftfreq(M))  # Horizontal U
    freq_y = np.fft.fftshift(np.fft.fftfreq(N))  # Vertical V
    fft_data = np.fft.fft2(image_data)
    fft_data_shifted = np.fft.fftshift(fft_data)  # Shift so that (0,0) is at center


def load_fits_file(path):
    # Open the FITS file
    global image_data
    #hdu_list = fits.open('./data/M87_EHT_2018_3644_b3.fits')
    try:
        hdu_list = fits.open(path)
    except:
        print("Could not load FITS file: {path}")
    # Access the primary HDU
    image_data = hdu_list[0].data.squeeze()

    # Close the FITS file
    hdu_list.close()




def open_fits_selection_window(event):
    global fits_data
    initial_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = filedialog.askopenfilename(title="Select a FITS file",
                                           filetypes=[("FITS files", "*.fits")],initialdir=initial_dir)
    load_fits_file(file_path)
    update_data_state()
    clear_uv_selection(None)
    render_image_data()


def clear_uv_selection(event):
    global selected_uv_points
    selected_uv_points = {'x':[],'y':[]}
    render_uv_selection_plot()

def render_image_data():
    global fits_plot,image_data
    fits_plot.imshow(image_data)
    fits_plot.set_title('FITS Image')
    fits_plot.set_xlabel('X Pixel')
    fits_plot.set_ylabel('Y Pixel')
    fits_plot.figure.canvas.draw_idle()





def render_result_plot():
        
    # Convert to index positions in the frequency matrix
    fft_sampled = np.zeros_like(fft_data_shifted, dtype=complex)
    for u, v in zip(selected_uv_points['x'],selected_uv_points['y']):
        u_idx = (np.abs(freq_x - u)).argmin()  # Find closest U index
        v_idx = (np.abs(freq_y - v)).argmin()  # Find closest V index
        fft_sampled[v_idx, u_idx] = fft_data_shifted[v_idx, u_idx]

    # Plot the reconstruction
    fft_sampled_unshifted = np.fft.ifftshift(fft_sampled)  # Undo shift
    reconstructed_image = np.fft.ifft2(fft_sampled_unshifted).real  # Inverse transform
    result_plot.imshow(reconstructed_image)
    result_plot.set_title("Reconstruction from UV points")
    result_plot.figure.canvas.draw_idle()

def render_uv_selection_plot():
    uv_selection_plot.clear()
    uv_selection_plot.set_title('UV selection')
    uv_selection_plot.set_xlabel('U')
    uv_selection_plot.set_ylabel('V')
    uv_selection_plot.scatter(selected_uv_points['x'],selected_uv_points['y'],c='blue')
    uv_selection_plot.set_xlim(freq_x.min(),freq_x.max())
    uv_selection_plot.set_ylim(freq_y.min(),freq_y.max())
    uv_selection_plot.vlines(x=0,ymin=freq_y.min(),ymax=freq_y.max(),linestyles='--',colors='grey')
    uv_selection_plot.hlines(y=0,xmin=freq_x.min(),xmax=freq_x.max(),linestyles='--',colors='grey')
    uv_selection_plot.figure.canvas.draw_idle()
    
    render_result_plot()


def onclick(event:MouseEvent):
    if uv_selection_plot.in_axes(event):
        if event.xdata != None and event.ydata != None:
            min_distance = 1
            if len(selected_uv_points['x']) > 0:
                distances = np.sqrt((selected_uv_points['x'] - event.xdata)**2 + (selected_uv_points['y'] - event.ydata)**2)
                min_idx = np.argmin(distances)  # Get index of closest point
                min_distance = distances[min_idx]

            if min_distance < 0.01:
                selected_uv_points['x'].pop(min_idx)
                selected_uv_points['y'].pop(min_idx)
            
            else:
                selected_uv_points['x'].append(event.xdata)
                selected_uv_points['y'].append(event.ydata)
            render_uv_selection_plot()
            


fig = plt.figure(figsize=(15,10))
fig.canvas.manager.set_window_title('FITS UV selection reconstruction')
gs = gridspec.GridSpec(4, 4, height_ratios=[2,1,1, 1])  # 2 rows, 2 columns
uv_selection_plot: Axes= plt.subplot(gs[:2, 1:3]) 
fits_plot: Axes= plt.subplot(gs[2:,:2 ]) 
result_plot: Axes= plt.subplot(gs[2:, 2:]) 

clear_selection_button_axes = plt.subplot(gs[1, 3]) 
select_fits_button_axes = plt.subplot(gs[1, 0]) 

select_fits_button = Button(select_fits_button_axes, 'Select FITS file',color="gray")
select_fits_button.on_clicked(open_fits_selection_window)


clear_uv_button = Button(clear_selection_button_axes, 'Clear selection',color="gray")
clear_uv_button.on_clicked(clear_uv_selection)



load_fits_file(default_path)
update_data_state()


print(f"U range: {freq_x.min()} to {freq_x.max()}")
print(f"V range: {freq_y.min()} to {freq_y.max()}")


fig.subplots_adjust(wspace=0.3,hspace=.4)  # Adjust horizontal & vertical space

render_image_data()
render_uv_selection_plot()
    

cid = uv_selection_plot.figure.canvas.mpl_connect('button_press_event', onclick)


plt.show()