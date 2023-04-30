import math
from utils import get_json_image
import numpy as np

#function to get points and labels
def get_annotations(img_file, json_data):
    json = get_json_image(img_file, json_data)
    cicles = json['cardiac_cycles']
    final_points_x = []
    final_points_y = []
    labels = []
    for i in range(len(cicles)):
        final_points_x.append([])
        final_points_y.append([])
        labels.append([])
        points = cicles[i]['control_points']
        for point in points:
            try:
                if point['x'] not in final_points_x: #x points cannot be repeated
                    final_points_x[i].append(point['x'])
                    final_points_y[i].append(point['y'])
                    labels[i].append(point['type'])
            except:
                pass  
    return final_points_x, final_points_y, labels

#function to get points and labels
def get_splines(img_file, json_data):
    json = get_json_image(img_file, json_data)
    cicles = json['cardiac_cycles']
    final_spline_x = []
    final_spline_y = []
    for i in range(len(cicles)):
        final_spline_x.append([])
        final_spline_y.append([])
        try:
            final_spline_x[i] = final_spline_x[i] + cicles[i]['spline']['x'] 
            final_spline_y[i] = final_spline_y[i] + cicles[i]['spline']['y'] 
        except:
            pass  
    return final_spline_x, final_spline_y

#computes distance between two points
def dist(x_1, x_2, y_1, y_2):
    return math.dist([x_1, y_1], [x_2, y_2])

#selects the point out of a spline that is in the middle between point 1 and point 2
def select_point(spline_x, spline_y, x_1, x_2, y_1, y_2):
    try:
        prev = float('inf')
        idx = len(spline_x) // 2
        prev_idx = 0
        
        while True:
            p_x = spline_x[idx]
            p_y = spline_y[idx]
            #print(x_1, p_x, y_1, p_y)
            left = dist(x_1, p_x, y_1, p_y)
            right = dist(x_2, p_x, y_2, p_y)
            diff = left - right 
            if abs(diff) >= abs(prev):
                return prev_idx     
            elif diff < 0:
                prev_idx = idx
                idx += 1
            else:
                prev_idx = idx
                idx -= 1
                
            prev = diff
    except:
        return len(spline_x) // 2

#creates a kpts between two points
def create_kpt(xPrev,xNext, yPrev,yNext, interp_x, interp_y, minX):
    idx_floor = xPrev -minX
    idx_top = xNext - minX
    res = select_point(interp_x[idx_floor:idx_top], 
                   interp_y[idx_floor:idx_top], 
                  xPrev, 
                  xNext, 
                  yPrev, 
                  yNext) + idx_floor 
    return interp_x[res], interp_y[res]

#computes the kpts 
def compute_kpts(final_points_x, final_points_y, interp_x, interp_y, labels, desired_labels):
    kpts_x = []
    kpts_y = []
    kpts_labels = []
    desired_idx = 0 
    i = 0
    minX = min(final_points_x)

    while i < len(labels) :
        if labels[i] == desired_labels[desired_idx]:
            mid_points = 0
            kpts_x.append(final_points_x[i])
            kpts_y.append(final_points_y[i])
            kpts_labels.append(labels[i])

            i += 1
            desired_idx += 1 

            if i > len(final_points_x) - 1 or desired_idx > len(desired_labels) - 1:
                break
                
            #create a key point
            xKpt, yKpt = create_kpt(final_points_x[i-1],final_points_x[i],
                                   final_points_y[i-1],final_points_y[i],
                                   interp_x, interp_y, minX)
            kpts_x.append(xKpt)
            kpts_y.append(yKpt)
            kpts_labels.append("spline_point")

            if labels[i]!= desired_labels[desired_idx] and desired_labels[desired_idx] == "diastolic peak":
                xKpt2, yKpt2 = create_kpt(kpts_x[-2],kpts_x[-1],
                       kpts_y[-2],kpts_y[-1],
                       interp_x, interp_y, minX)
                xKpt3, yKpt3 = create_kpt(kpts_x[-1],final_points_x[i],
                       kpts_y[-1], final_points_y[i],
                       interp_x, interp_y, minX)
                kpts_x = kpts_x + [xKpt2, xKpt3]
                kpts_y = kpts_y + [yKpt2, yKpt3]
                kpts_labels = kpts_labels + ["spline_point", "spline_point"] 
                desired_idx += 1
        else:
            i += 1
    return kpts_x, kpts_y, kpts_labels

#scales the x kpts
def transform_x(x, cycle_metadata):
    return x - cycle_metadata["min_x"] + cycle_metadata["x_shift"]

#scales the y kpts
def transform_y(y, cycle_metadata):
    return y - cycle_metadata["min_y"] + cycle_metadata["y_shift"]

def transform_list(input_list, cycle_metadata, axis="x"):
    output_list = []
    function = transform_x if axis == "x" else transform_y
    for kpt in input_list:
        output_list.append(function(kpt, cycle_metadata))
    return output_list

#scales the x kpts
def detransform_x(x, cycle_metadata):
    return x + cycle_metadata["min_x"] - cycle_metadata["x_shift"]

#scales the y kpts
def detransform_y(y, cycle_metadata):
    return y + cycle_metadata["min_y"] - cycle_metadata["y_shift"]

def detransform_list(input_list, cycle_metadata, axis="x"):
    output_list = []
    function = detransform_x if axis == "x" else detransform_y
    for kpt in input_list:
        output_list.append(function(kpt, cycle_metadata))
    return output_list