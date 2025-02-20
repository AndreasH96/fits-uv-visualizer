# FITS file reconstructor visualizer
A *very* simple app for me to learn more about how UV coverage works for a given FITS file. 

Allows you to load a fits file, select UV points and see how the UV coverage affects the reconstruction of the original image. 


![single_img](./imgs/generate_N_points.gif)

## How to run
1. Clone the repository 
~~~~
git clone https://github.com/AndreasH96/fits-uv-visualizer.git 
cd ./fits-uv-visualizer
~~~~
2. Install requirements
~~~~
pip install -r requirements.txt
~~~~

3. Run the application
~~~~
python app.py
~~~~

## Example: Single point selected
![single_img](./imgs/single_point_selected.png)

## Example: Multiple points selected
![multipe_img](./imgs/multiple_points_selected.png)

In the example with multiple points selected, we can see that the actual image (point source, nothing fancy) can be somewhat seen from the reconstruction.


## Example: Placements of UV points (M87)
### 1000 randomly placed points:

![m87_rand_spread](./imgs/m87_spread_out.png)

### 1000 randomly placed points, with a maximum UV distance of 0.05 (to focus on capturing general aspects):

![m87_rand_centered](./imgs/m87_centered.png)

# Todo:
- [x] Generate N random UV points with min and max UV distance
- [ ] Phase shifting ?
- [ ] Load file of UV points
- [ ] Evaluate UV ranges, should allow other than the fixed -0.5 - 0.5?