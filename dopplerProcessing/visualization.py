from matplotlib import figure, pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from utils import *
import cv2 
from dopplerProcessing.rkt_spline_module import generateSpline
import scipy.interpolate as interpolate
import numpy as np

## DRAW IMAGE
# def draw_image(img_path):
#     pixel_array_rgb = load_image(img_path)
#     if len(pixel_array_rgb.shape) == 4:
#         plt.imshow(pixel_array_rgb[0], interpolation='spline36')
#     else:
#         plt.imshow(pixel_array_rgb, interpolation='spline36')

#draw image interactive
def draw_img_interactive(pixel_array):
    return px.imshow(pixel_array)

#draw points 
def draw_points_interactive(final_points_x, final_points_y, labels, fig):    
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=final_points_x, 
            y=final_points_y,
            hovertext= labels,
            line_color="red",
            line_width=2,
            showlegend=False
        )
    )

def draw_kpts(img, kpts, color = [0,255,0]):
    im = img.copy() # workaround a bug in Python OpenCV wrapper: https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    # draw points
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=color)
    return im

def plot_kpts_pred_and_gt(fig, img, gt_kpts=None, pred_kpts=None, kpts_info=[], closed_contour=False):

    fig.clf()
    clean_img = img

    interpolation_step_size = 0.001  # 0.01
    unew = np.arange(0, 1.00, interpolation_step_size)

    if gt_kpts is not None:
        img = draw_kpts(img, gt_kpts, color=[0,255,0])
        try:
            gt_interpolate = generateSpline(gt_kpts[:, 0].astype("int"), gt_kpts[:, 1].astype("int"))
        except Exception as e:
            gt_tck, _ = interpolate.splprep([gt_kpts[:, 0], gt_kpts[:, 1]], s=0)
            gt_interpolate = interpolate.splev(unew, gt_tck)
    
    if pred_kpts is not None:
        img = draw_kpts(img, pred_kpts, color=[255,0,0])
        try:
            pred_interpolate = generateSpline(pred_kpts[:, 0].astype("int"), pred_kpts[:, 1].astype("int"))
        except Exception as e:
            print(e)
            pred_tck, _ = interpolate.splprep([pred_kpts[:, 0], pred_kpts[:, 1]], s=0)
            pred_interpolate = interpolate.splev(unew, pred_tck)

    # option 1: clean img + kpts_img
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(clean_img)
    ax.set_axis_off()
    
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(clean_img)
    if gt_kpts is not None:
        ax.plot(gt_interpolate[0], gt_interpolate[1], marker=None, linestyle = '-' , c='green')
    if pred_kpts is not None:
        ax.plot(pred_interpolate[0], pred_interpolate[1], marker=None, linestyle = '-', c='red') # fixme: change color back to 'white'
    ax.set_axis_off()
    
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img)

    return fig