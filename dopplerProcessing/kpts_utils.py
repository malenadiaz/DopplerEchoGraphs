import math
import numpy as np
from dopplerProcessing.rkt_spline_module import generateSpline

#function to get points and labels
def get_annotations(json):
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
def get_splines(json):
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
def create_kpt(xPrev,xNext, yPrev,yNext, interp_x, interp_y, prev_idx_top):
    idx_floor = prev_idx_top
    idx_top = prev_idx_top
    for val in interp_x[idx_floor:]:
        if val < xNext:
            idx_top += 1
    res = select_point(interp_x[idx_floor:idx_top], 
                   interp_y[idx_floor:idx_top], 
                  xPrev, 
                  xNext, 
                  yPrev, 
                  yNext) + idx_floor 
    return interp_x[res], interp_y[res], idx_top, idx_floor

#computes the kpts 
def compute_kpts(final_points_x, final_points_y, interp_x, interp_y, labels, desired_labels):
    kpts_x = []
    kpts_y = []
    kpts_labels = []
    desired_idx = 0 
    i = 0
    curr_top = 0

    while i < len(labels) :
        if labels[i] == desired_labels[desired_idx]:
            mid_points = 0

            kpts_x.append(final_points_x[i])
            kpts_y.append(final_points_y[i])
            kpts_labels.append(labels[i])
            
            prev = i
            i += 1
            desired_idx += 1 

            if i > len(final_points_x) - 1 or desired_idx > len(desired_labels) - 1:
                break

            while labels[i] == 'mid_control_point' or labels[i] == 'mid deceleration point': #ignore mid control points
                i += 1

            #create a key point
            xKpt, yKpt, curr_top, curr_floor = create_kpt(final_points_x[prev],final_points_x[i],
                                   final_points_y[prev],final_points_y[i],
                                   interp_x, interp_y, curr_top)
            kpts_x.append(xKpt)
            kpts_y.append(yKpt)
            kpts_labels.append("spline_point")

            if (labels[i]!= desired_labels[desired_idx] and desired_labels[desired_idx] == "diastolic peak") or labels[i] == 'ejection end':
                
                xKpt2, yKpt2, curr_top, curr_floor = create_kpt(kpts_x[-2],kpts_x[-1],
                       kpts_y[-2],kpts_y[-1],
                       interp_x, interp_y, curr_floor)
                kpts_x.insert(-1, xKpt2)
                kpts_y.insert(-1 ,yKpt2 )
                xKpt3, yKpt3, curr_top, curr_floor = create_kpt(kpts_x[-1],final_points_x[i],
                       kpts_y[-1], final_points_y[i],
                       interp_x, interp_y, curr_top)
                kpts_x.append(xKpt3)
                kpts_y.append(yKpt3)
                kpts_labels = kpts_labels + ["spline_point", "spline_point"] 
                desired_idx += 1
        else:
            i += 1
    return kpts_x, kpts_y, kpts_labels



#scales the x kpts
def transform_x(x, cycle_metadata):
    temp =  x - cycle_metadata["min_x"] + cycle_metadata["margin"]
    return int(temp*cycle_metadata["ratio"] + cycle_metadata["x_shift"])

#scales the y kpts
def transform_y(y, cycle_metadata):
    temp =  y - cycle_metadata["min_y"] + cycle_metadata["margin"]
    return int(temp*cycle_metadata["ratio"]  + cycle_metadata["y_shift"])

def transform_list(input_list, cycle_metadata, axis="x"):
    output_list = []
    function = transform_x if axis == "x" else transform_y
    for kpt in input_list:
        output_list.append(function(kpt, cycle_metadata))
    return output_list

#scales the x kpts
def detransform_x(x, cycle_metadata):
    temp = (x - cycle_metadata["x_shift"]) // cycle_metadata["ratio"]
    return temp - cycle_metadata["margin"] + cycle_metadata["min_x"] 

#scales the y kpts
def detransform_y(y, cycle_metadata):
    temp = (y - cycle_metadata["y_shift"]) // cycle_metadata["ratio"]
    return temp - cycle_metadata["margin"] + cycle_metadata["min_y"] 

def detransform_list(input_list, cycle_metadata):
    output_list = []
    for kpt in input_list:
        x = detransform_x(kpt[0], cycle_metadata)
        y = detransform_y(kpt[1], cycle_metadata)
        output_list.append([x,y])
    return output_list

def to_physical_x(x, metadata):
    res =  (x - metadata["min_x"])  * metadata["physical_delta_x"]
    return res

def to_physical_y(y, metadata):
    return np.abs((metadata["zero_line"] - (y - metadata["min_y"])) * metadata["physical_delta_y"])

def to_physical_list(input_list, metadata ):
    output_list = []
    for kpt in input_list:
        x = to_physical_x(kpt[0], metadata)
        y = to_physical_y(kpt[1], metadata)
        output_list.append([x,y])
    return output_list

def save_kpts(kpts_x, kpts_y, kpts_labels, output_dir, output_name,folder, NUM_KPTS, DIMS):
    if any(x > DIMS[0] and x < 0 for x in kpts_x ) or  any(y > DIMS[1] and y < 0 for y in kpts_y ):
        print("Error in" + output_name + " in key points construction.")
        return False
    #save the kpts if everything went well 
    if len(kpts_x)!= NUM_KPTS or len(kpts_labels)!= NUM_KPTS or len(kpts_y)!= NUM_KPTS:
        print("Image" + output_name + "has more or less keypoints than needed.")
        return False
    else:
        np.save( output_dir + folder + output_name + '.npy', 
                np.column_stack((np.column_stack((kpts_x, kpts_y)), kpts_labels)))
    return True

#computes the kpts 
def compute_kpts_wo_added(final_points_x, final_points_y, interp_x, interp_y, labels, desired_labels):
    kpts_x = []
    kpts_y = []
    kpts_labels = []
    desired_idx = 0 
    i = 0
    curr_top = 0

    while i < len(labels) :
        if labels[i] == desired_labels[desired_idx]:
            mid_points = 0
            kpts_x.append(final_points_x[i])
            kpts_y.append(final_points_y[i])
            kpts_labels.append(labels[i])
            
            prev = i
            i += 1
            desired_idx += 1 

            if i > len(final_points_x) - 1 or desired_idx > len(desired_labels) - 1:
                break
            if (labels[i]!= desired_labels[desired_idx] and desired_labels[desired_idx] == "diastolic peak"):
                floor = np.where(np.array(interp_x) == final_points_x[i-1])[0][0]
                xKpt2, yKpt2, curr_top, curr_floor = create_kpt(final_points_x[i-1],final_points_x[i],
                       final_points_y[i-1],final_points_y[i],
                       interp_x, interp_y, floor)
                kpts_x = kpts_x + [xKpt2]
                kpts_y = kpts_y + [yKpt2]
                kpts_labels = kpts_labels + ["spline_point"] 
                desired_idx += 1
        else:
            i += 1
    return kpts_x, kpts_y, kpts_labels

def diff2real(kpts):
    new_x = np.cumsum(kpts[:,0])
    new_kpts = np.column_stack((new_x, kpts[:,1]))
    return new_kpts


def pardiff2real(kpts, labels):
    new_kpts = []
    last_idx = 0
    kpts_x = kpts[:,0]
    kpts_y = kpts[:,1]

    sp_points = [] 
    sp_points_y = []
    for i in range(len(kpts_x)):
        if labels[i] not in ["ejection beginning", "maximum velocity", "ejection end"]:
            sp_points.append(kpts_x[i] + kpts_x[last_idx])
            sp_points_y.append(kpts_y[i])
        else:
            if sp_points != []:
                new_kpts += np.column_stack((sp_points, sp_points_y)).tolist()
                sp_points = []
                sp_points_y = []
                last_idx = i
            new_kpts.append([kpts_x[i], kpts_y[i]])
    return np.array(new_kpts)

def compute_splines(gt_kpts, pred_kpts, img_path):
    gt_interpolate, pred_interpolate = None,None
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
        try:
            gt_interpolate = generateSpline(gt_kpts[:,0], gt_kpts[:,1])
        except:
            print(gt_kpts[:,0], gt_kpts[:,1], "\n")

    if pred_kpts is not None:
        try:
            pred_interpolate = generateSpline(pred_kpts[:,0], pred_kpts[:,1])
        except:
            print(pred_kpts[:,0], pred_kpts[:,1], "\n")

    return gt_interpolate, pred_interpolate

def list_compute_spline(all_pred_kpts, all_gt_kpts, img_paths):
    gt_spline_list = []
    pred_spline_list = []
    valid_pos = []
    for i in range(len(all_pred_kpts[0])):
        gt_spline, pred_spline = compute_splines(all_gt_kpts[i], all_pred_kpts[i], img_paths[i])
        if gt_spline is not None and pred_spline is not None:
            valid_pos.append(i)
            gt_spline_list.append(gt_spline)
            pred_spline_list.append(pred_spline)
    return gt_spline_list, pred_spline_list, valid_pos