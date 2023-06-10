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

def draw_kpts_plt(kpts_x, kpts_y, ax):
    ax.plot(kpts_x, kpts_y,"ro")
    return ax

def save_img_annotated(pix, kpts_x, kpts_y, output_dir):
    fig = plt.gcf()
    fig.clf()
    ax = plt.gca()
    plt.imshow(pix, interpolation='spline36')
    ax = draw_kpts_plt(kpts_x, kpts_y, ax)
    plt.savefig(output_dir)

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

def plot_legend(labels, colors):
    
    width, height = 200, 300

    # Create a white background image
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    legend.fill(255)

    # Define the font and font scale
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5

    # Calculate the position and size of each label box
    box_width = 10
    box_height = 10
    box_positions = [(0, i * (box_height + 10) + 10) for i in range(len(labels)+2)]

    
    cv2.circle(legend, (box_positions[0][0] + 5, box_positions[0][1] + 5), 3, (0,0,0), -1)
    cv2.putText(legend, "Predictions", (box_positions[0][0] + 10, box_positions[0][1] + box_height), font, font_scale, (0,0,0), 1, cv2.LINE_AA)
    cv2.rectangle(legend, box_positions[1], ( box_positions[1][0] + 7 ,box_positions[1][1] + 7), (0,0,0), 1)
    cv2.putText(legend, "GT values", (box_positions[1][0] + 10, box_positions[1][1] + box_height), font, font_scale, (0,0,0), 1, cv2.LINE_AA)

    # Draw the label boxes with their respective colors and labels
    for i, (label, color, position) in enumerate(zip(labels, colors, box_positions[2:])):
        cv2.rectangle(legend, position, (position[0] + box_width, position[1] + box_height), color, -1)
        label_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        label_position = (position[0] + (box_width - label_size[0]) // 2, position[1] + (box_height + label_size[1]) // 2)
        cv2.putText(legend, label, (box_width + 10, position[1] + box_height), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return legend

def draw_kpts(img, kpts, mode = 'gt', colors=None):
    im = img.copy() # workaround a bug in Python OpenCV wrapper: https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    # draw points
    for i in range(len(kpts)):
        k = kpts[i]
        x = int(k[0])
        y = int(k[1])
        if mode == 'gt':
            im = cv2.rectangle(im, (x-2, y-2), (x+2, y+2), colors[i], 1) 
        else:
            im = cv2.circle(im, (x, y), radius=3, thickness=-1, color=colors[i])
    
    return im

def plot_kpts_pred_and_gt(fig, img, gt_kpts=None, pred_kpts=None, kpts_info=None, img_path=None, colors = None):
    
    fig.clf()
    clean_img = img

    interpolation_step_size = 0.001  # 0.01
    unew = np.arange(0, 1.00, interpolation_step_size)

    #pos =  [i for i, s in enumerate(kpts_info) if 'spline_point' not in s]
    #pos = range(0, gt_kpts.shape[0], 2)
    gt_kpts_all = gt_kpts
    pred_kpts_all = pred_kpts
    if len(gt_kpts) > 7:
        pos = list(range(0,13,2))
    else:
        pos = list(range(7))
    gt_kpts = gt_kpts[pos, :].astype("int")
    pred_kpts = pred_kpts[pos, : ].astype("int")

    # Iterate through the array
    for i in range(len(gt_kpts) - 1):
        if gt_kpts[i][0] == gt_kpts[i + 1][0]:
            gt_kpts[i + 1][0] += 1
    for i in range(len(pred_kpts) - 1):
        if pred_kpts[i][0] == pred_kpts[i + 1][0]:
            pred_kpts[i + 1][0] += 1

    if gt_kpts is not None:
        img = draw_kpts(img, gt_kpts_all, mode='gt', colors=colors)

        try:
            gt_interpolate = generateSpline(gt_kpts[:,0], gt_kpts[:,1])
        except:
            print("Exception generating gt spline for image {}:{}".format(img_path, e))
            gt_tck, _ = interpolate.splprep([gt_kpts[pos, 0], gt_kpts[pos, 1]], s=0)
            gt_interpolate = interpolate.splev(unew, gt_tck)

    if pred_kpts is not None:
        img = draw_kpts(img, pred_kpts_all, mode='pred', colors=colors)
        try:
            pred_interpolate = generateSpline(pred_kpts[:,0], pred_kpts[:,1])
        except:
            print("Exception generating pred spline for image {}:{}".format(img_path, e))
            pred_tck, _ = interpolate.splprep([pred_kpts[pos, 0], pred_kpts[pos, 1]], s=0)
            pred_interpolate = interpolate.splev(unew, pred_tck)

    # # option 1: clean img + kpts_img
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(clean_img)
    # ax.set_axis_off()

    specs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[3,3,1])

    #plot two figures 
    ax = fig.add_subplot(specs[0])
    ax.imshow(clean_img)
    if gt_kpts is not None:
        ax.plot(gt_interpolate[0], gt_interpolate[1], marker=None, linestyle = '-' , c='green', label='GT spline',)
    if pred_kpts is not None:
        ax.plot(pred_interpolate[0], pred_interpolate[1], marker=None, linestyle = '-', c='red', label='PRED spline',) # fixme: change color back to 'white'
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.set_axis_off()
    
    ax = fig.add_subplot(specs[1])
    ax.imshow(img)
    ax.set_axis_off()

    ax = fig.add_subplot(specs[2])
    ax.imshow(plot_legend(kpts_info, colors))
    ax.set_axis_off()

    return fig