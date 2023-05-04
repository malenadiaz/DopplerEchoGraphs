import pickle
import json
from pydicom import dcmread
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot  as plt
import plotly.express as px
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm
import os 

def pulsatily_idx(kpts):
    return (max(kpts) - min(kpts)) / np.mean(kpts)

def get_pulsatily_idx(pred_kpts, gt_kpts, err_metric):
    pred_pul_idx = np.apply_along_axis(pulsatily_idx, 1, pred_kpts[:,:,1])
    gt_pul_idx = np.apply_along_axis(pulsatily_idx, 1, gt_kpts[:,:,1])
    return err_metric(pred_pul_idx, gt_pul_idx)

def get_pulsatility_idx_point(pred_kpts, gt_kpts, err_metric):
    pred_pul_idx = np.apply_along_axis(pulsatily_idx, 1, pred_kpts[:,:,1] )
    gt_pul_idx = np.apply_along_axis(pulsatily_idx, 1, gt_kpts[:,:,1])
    res = []
    for i in range(len(pred_kpts)):
        res.append(err_metric([gt_pul_idx[i]],[pred_pul_idx[i]]))
    return res

def ejection_time(kpts, beg_idx, end_idx):
    return kpts[end_idx] - kpts[beg_idx]

def get_ejection_time(pred_kpts, gt_kpts, labels, err_metric):
    beg_index = next((index for index, label in enumerate(labels) if label  == 'ejection beginning'), -1)
    end_index = next((index for index, label in enumerate(labels) if label  == 'maximum velocity'), -1)
    pred_pul_idx = np.apply_along_axis(ejection_time, 1, pred_kpts[:,:,0],beg_index, end_index)
    gt_pul_idx = np.apply_along_axis(ejection_time, 1,gt_kpts[:,:,0], beg_index, end_index)
    return err_metric(pred_pul_idx, gt_pul_idx)

def get_ejection_time_point(pred_kpts, gt_kpts, labels, err_metric):
    beg_index = next((index for index, label in enumerate(labels) if label  == 'ejection beginning'), -1)
    end_index = next((index for index, label in enumerate(labels) if label  == 'maximum velocity'), -1)
    pred_eje_time = np.apply_along_axis(ejection_time, 1, pred_kpts[:,:,0],beg_index, end_index)
    gt_eje_time  = np.apply_along_axis(ejection_time, 1,gt_kpts[:,:,0], beg_index, end_index)
    res = []
    for i in range(len(pred_kpts)):
        res.append(err_metric([gt_eje_time[i]],[pred_eje_time[i]]))
    return res

def get_metric(pred_kpts, gt_kpts, metric):
    res = []
    res.append(metric(gt_kpts.flatten(), pred_kpts.flatten()))
    for i in range(len(pred_kpts[0])):
        res.append(metric( gt_kpts[:,i], pred_kpts[:,i]))
    return res

def compute_mse(pred_kpts, gt_kpts):
    return mean_squared_error(gt_kpts, pred_kpts)

def compute_kpts_err(pred_kpts, gt_kpts):
    return mean_squared_error(gt_kpts.flatten(), pred_kpts.flatten(),)


def load_pickle(path):
    with open(path,'rb') as fp:
        p = pickle.load(fp)
    pred_kpts = np.array([img['keypoints_prediction'] for img in p.values()])
    gt_kpts = np.array([img['keypoints'] for img in p.values()])
    return pred_kpts, gt_kpts 

def metric_by_point(pred_kpts, gt_kpts, metric):
    total = []
    x =  []
    y = []
    for i in range(len(pred_kpts)):
        total.append(metric(gt_kpts[i,:,:], pred_kpts[i,:,:]))
        x.append(metric(gt_kpts[i,:,0], pred_kpts[i,:,0]))
        y.append( metric(gt_kpts[i,:,1], pred_kpts[i,:,1]))
    return total, x, y 

def stats_report_point_doppler(pred_kpts, gt_kpts, phys_pred_kpts, phys_gt_kpts, labels, dict_keys):
    df = pd.DataFrame(columns = ["PIX MAPE", "PIX MAPE X", "PIX MAPE Y"
                                 ,"PHYS MAPE", "PHYS MAPE X", "PHYS MAPE Y"
                                 ,"PIX MSE", "PIX MSE X", "PIX MSE Y"
                                 ,"PHYS MSE", "PHYS MSE X", "PHYS MSE Y"
                                 ,"PUL IDX MAPE", "PUL IDX MSE",
                                 "EJE TIME MAPE", "EJE TIME MSE"], index = dict_keys)
    df["PIX MAPE"], df["PIX MAPE X"], df["PIX MAPE Y"] = metric_by_point(pred_kpts, gt_kpts, mean_absolute_percentage_error)
    df["PIX MSE"], df["PIX MSE X"], df["PIX MSE Y"] = metric_by_point(pred_kpts, gt_kpts, mean_squared_error)
    df["PHYS MAPE"], df["PHYS MAPE X"], df["PHYS MAPE Y"] = metric_by_point(phys_pred_kpts, phys_gt_kpts, mean_absolute_percentage_error)
    df["PHYS MSE"], df["PHYS MSE X"], df["PHYS MSE Y"] = metric_by_point(phys_pred_kpts, phys_gt_kpts, mean_squared_error)
    df["PUL IDX MAPE"] = get_pulsatility_idx_point(pred_kpts, gt_kpts, mean_absolute_percentage_error)
    df["PUL IDX MSE"] = get_pulsatility_idx_point(pred_kpts, gt_kpts, mean_squared_error)
    df["EJE TIME MAPE"] = get_ejection_time_point(pred_kpts, gt_kpts, labels, mean_absolute_percentage_error)
    df["EJE TIME MSE"] = get_ejection_time_point(pred_kpts, gt_kpts, labels, mean_squared_error)


    return df


def create_point_annotations_excel(pred_kpts, gt_kpts, phys_pred_kpts, phys_gt_kpts, labels, dict_keys, output_dir):
    columns= []
    for lbl in labels:
        columns.append(lbl + '_REAL_X')
        columns.append(lbl + '_PREDICTED_X')
        columns.append(lbl + '_REAL_Y')
        columns.append(lbl + '_PREDICTED_Y')

    df_1 = pd.DataFrame(columns = columns , index = dict_keys)
    df_2 = pd.DataFrame(columns = columns , index = dict_keys)
    df_3 = pd.DataFrame(columns = columns , index = dict_keys)
    df_4 = pd.DataFrame(columns = columns , index = dict_keys)

    for i in range(len(labels)):
        df_1[labels[i] + '_REAL_X'] = gt_kpts[:,i, 0]
        df_1[labels[i] + '_PREDICTED_X'] = pred_kpts[:,i, 0]
        df_3[labels[i] + '_ERROR_X'] = np.abs(gt_kpts[:,i, 0] - pred_kpts[:,i, 0])
        df_1[labels[i] + '_REAL_Y'] = gt_kpts[:,i, 1]
        df_1[labels[i] + '_PREDICTED_Y'] = pred_kpts[:,i, 1]
        df_3[labels[i] + '_ERROR_Y'] = np.abs(gt_kpts[:,i, 1] - pred_kpts[:,i, 1])
        df_2[labels[i] + '_REAL_X'] = phys_gt_kpts[:,i, 0]
        df_2[labels[i] + '_PREDICTED_X'] = phys_pred_kpts[:,i, 0]
        df_4[labels[i] + '_ERROR_X'] = np.abs(phys_gt_kpts[:,i, 0] - phys_pred_kpts[:,i, 0])
        df_2[labels[i] + '_REAL_Y'] = phys_gt_kpts[:,i, 1]
        df_2[labels[i] + '_PREDICTED_Y'] = phys_pred_kpts[:,i, 1]
        df_4[labels[i] + '_ERROR_Y'] = np.abs(phys_gt_kpts[:,i, 1] - phys_pred_kpts[:,i, 1])

    with pd.ExcelWriter(os.path.join(output_dir, "real_vs_pred.xlsx")) as writer:  

        df_1.to_excel(writer, sheet_name='Pixel')
        df_2.to_excel(writer, sheet_name='Physical')
        df_3.to_excel(writer, sheet_name='Pixel_error')
        df_4.to_excel(writer, sheet_name='Physical_error')

def compute_stats_original_points(pred_kpts, gt_kpts, phys_pred_kpts, phys_gt_kpts, labels):
    og_idxs = [index for index, value in enumerate(labels) if value != 'spline_point']
    og_lbls = [value for value in labels if value != 'spline_point']

    return stats_report_doppler(pred_kpts[:,og_idxs,:], gt_kpts[:,og_idxs,:], phys_pred_kpts[:,og_idxs,:], phys_gt_kpts[:,og_idxs,:], og_lbls)

def stats_report_doppler(pred_kpts, gt_kpts, phys_pred_kpts, phys_gt_kpts, labels):
    pf = pd.DataFrame(columns= ['TOTAL'] + labels ,
                       index=["PIX MAPE", "PIX MAPE X", "PIX MAPE Y" 
                            ,"PIX MSE", "PIX MSE X", "PIX MSE Y" 
                            ,"PHYS MAPE", "PHYS MAPE X", "PHYS MAPE Y"
                            ,"PHYS MSE", "PHYS MSE X","PHYS MSE Y" 
                            ,"PUL IDX MAPE", "PUL IDX MSE"
                            ,"EJE TIME MAPE", "EJE TIME MSE"])
    
    pf.loc["PIX MAPE"] = get_metric(pred_kpts, gt_kpts, mean_absolute_percentage_error)
    pf.loc["PIX MAPE X"] = get_metric(pred_kpts,gt_kpts,  mean_absolute_percentage_error)
    pf.loc["PIX MAPE Y"] = get_metric(pred_kpts,gt_kpts,  mean_absolute_percentage_error)

    pf.loc["PIX MSE"] = get_metric( pred_kpts, gt_kpts, mean_squared_error)
    pf.loc["PIX MSE X"] = get_metric( pred_kpts[:,:,0], gt_kpts[:,:,0], mean_squared_error)
    pf.loc["PIX MSE Y"] = get_metric( pred_kpts[:,:,1], gt_kpts[:,:,1], mean_squared_error)

    pf.loc["PHYS MAPE"] = get_metric(phys_pred_kpts, phys_gt_kpts, mean_absolute_percentage_error)
    pf.loc["PHYS MAPE X"] = get_metric( phys_pred_kpts[:,:,0],phys_gt_kpts[:,:,0], mean_absolute_percentage_error)
    pf.loc["PHYS MAPE Y"] = get_metric( phys_pred_kpts[:,:,1],phys_gt_kpts[:,:,1], mean_absolute_percentage_error)

    pf.loc["PHYS MSE"] = get_metric(phys_pred_kpts, phys_gt_kpts, mean_squared_error)
    pf.loc["PHYS MSE X"] = get_metric(phys_pred_kpts[:,:,0], phys_gt_kpts[:,:,0], mean_squared_error)
    pf.loc["PHYS MSE Y"] = get_metric(phys_pred_kpts[:,:,1],phys_gt_kpts[:,:,1],  mean_squared_error)

    pf.loc[["PUL IDX MAPE"],["TOTAL"]] = get_pulsatily_idx(phys_pred_kpts, phys_gt_kpts, mean_absolute_percentage_error)
    pf.loc[["PUL IDX MSE"],["TOTAL"]] = get_pulsatily_idx(phys_pred_kpts, phys_gt_kpts, mean_squared_error)
   
    pf.loc[["EJE TIME MAPE"],["TOTAL"]] = get_ejection_time(phys_pred_kpts, phys_gt_kpts,labels, mean_absolute_percentage_error)
    pf.loc[["EJE TIME MSE"],["TOTAL"]] = get_ejection_time(phys_pred_kpts, phys_gt_kpts,labels, mean_squared_error)
    
    pf = pf.astype(float)

    # idxs = np.argwhere(phys_gt_kpts == 0 )
    # print(idxs)

    return pf

def create_bland_altman(all_pred_kpts,all_gt_kpts, all_phys_pred_kpts, all_phys_gt_kpts,labels, output_dir):    
    
    f, ax = plt.subplots(1,2, figsize = (25, 20))
    sm.graphics.mean_diff_plot(all_gt_kpts.flatten(), all_pred_kpts.flatten(), ax = ax[0])
    ax[0].set_title("All points pixel-wise")
    sm.graphics.mean_diff_plot(all_phys_gt_kpts.flatten(), all_phys_pred_kpts.flatten(), ax = ax[1])
    ax[1].set_title("All points physical")
    f.savefig(os.path.join(output_dir,'ALL_points.png'))

    f.clf()
    f, ax = plt.subplots(1,2, figsize = (25, 20))
    sm.graphics.mean_diff_plot(all_gt_kpts[:,:,0].flatten(), all_pred_kpts[:,:,0].flatten(), ax = ax[0])
    ax[0].set_title("X points pixel-wise")
    sm.graphics.mean_diff_plot(all_gt_kpts[:,:,1].flatten(), all_pred_kpts[:,:,1].flatten(), ax = ax[1])
    ax[1].set_title("Y points pixel-wise")
    f.savefig(os.path.join(output_dir,'XY_points_pixel.png'))
 
    f.clf()
    f, ax = plt.subplots(1,2, figsize = (25, 20))
    sm.graphics.mean_diff_plot(all_phys_gt_kpts[:,:,0].flatten(), all_phys_pred_kpts[:,:,0].flatten(), ax = ax[0])
    ax[0].set_title("X points physical")
    sm.graphics.mean_diff_plot(all_phys_gt_kpts[:,:,1].flatten(), all_phys_pred_kpts[:,:,1].flatten(), ax = ax[1])
    ax[1].set_title("X points physical")
    f.savefig(os.path.join(output_dir,'XY_points_physical.png'))

    ##########
    
    num_kpts = len(all_pred_kpts[0])

    for i in range(num_kpts):
        f.clf()
        f, ax = plt.subplots(nrows=2, ncols=3, figsize=(25,25))

        sm.graphics.mean_diff_plot(all_gt_kpts[:,i,:].flatten(), all_pred_kpts[:,i,:].flatten(), ax = ax[0][0])
        ax[0][0].set_title("kpt_{} pixel x and y".format(i))
        sm.graphics.mean_diff_plot(all_gt_kpts[:,i,0].flatten(), all_pred_kpts[:,i,0].flatten(), ax = ax[0][1])
        ax[0][1].set_title("kpt_{} pixel x".format(i))
        sm.graphics.mean_diff_plot(all_gt_kpts[:,i,1].flatten(), all_pred_kpts[:,i,1].flatten(), ax = ax[0][2])
        ax[0][2].set_title("kpt_{} pixel y".format(i))
        sm.graphics.mean_diff_plot(all_phys_gt_kpts[:,i,:].flatten(), all_phys_pred_kpts[:,i,:].flatten(), ax = ax[1][0])
        ax[1][0].set_title("kpt_{} physical x and y".format(i))
        sm.graphics.mean_diff_plot(all_phys_gt_kpts[:,i,0].flatten(), all_phys_pred_kpts[:,i,0].flatten(), ax = ax[1][1])
        ax[1][1].set_title("kpt_{} physical x".format(i))
        sm.graphics.mean_diff_plot(all_phys_gt_kpts[:,i,1].flatten(), all_phys_pred_kpts[:,i,1].flatten(), ax = ax[1][2])
        ax[1][2].set_title("kpt_{} physical y".format(i))

        f.savefig(os.path.join(output_dir,'{}_point_.png'.format(i)))

