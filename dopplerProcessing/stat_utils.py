import pickle
import json
from pydicom import dcmread
import cv2
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot  as plt
import plotly.express as px
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, auc
import statsmodels.api as sm
import os 
from dopplerProcessing.rkt_spline_module import generateSpline

def mse_error_2dim(gt, preds):
    res =  np.sum((preds[:,0] - gt[:,0])**2 + (preds[:,1] - gt[:,1])**2)
    res = res/preds.shape[0]
    res = np.sqrt(res)
    return res

def pulsatily_idx(kpts):
    return (max(kpts) - min(kpts)) / np.mean(kpts)

def get_pulsatily_idx(pred_kpts, gt_kpts):
    pred_pul_idx = np.array([pulsatily_idx(pred[:,1]) for pred in pred_kpts])
    gt_pul_idx = np.array([pulsatily_idx(gt[:,1]) for gt in gt_kpts])
    if np.any(pred_pul_idx<0) or np.any(gt_pul_idx<0):
        print('PUL IDX IS NEGATIVE', np.mean(pred_kpts[:,:,1].flatten()), np.mean(gt_kpts[:,:,1].flatten()) )
    
    return pred_pul_idx, gt_pul_idx

def get_pulsatility_idx_point(pred_kpts, gt_kpts, err_metric):
    pred_pul_idx = np.apply_along_axis(pulsatily_idx, 1, pred_kpts[:,:,1] )
    gt_pul_idx = np.apply_along_axis(pulsatily_idx, 1, gt_kpts[:,:,1])
    res = []
    for i in range(len(pred_kpts)):
        res.append(err_metric([gt_pul_idx[i]],[pred_pul_idx[i]]))
    return res

def ejection_time(kpts, beg_idx, end_idx):
    return kpts[end_idx] - kpts[beg_idx]

def get_ejection_time(pred_kpts, gt_kpts, labels):
    beg_index = next((index for index, label in enumerate(labels) if label  == 'ejection beginning'), -1)
    end_index = next((index for index, label in enumerate(labels) if label  == 'maximum velocity'), -1)
    pred_eje_time = np.apply_along_axis(ejection_time, 1, pred_kpts[:,:,0],beg_index, end_index)
    gt_eje_time = np.apply_along_axis(ejection_time, 1,gt_kpts[:,:,0], beg_index, end_index)
    return  pred_eje_time, gt_eje_time

def get_vti(pred_splines, gt_splines):
    pred_vti = np.array([auc(spline[:,0], spline[:,1]) for spline in pred_splines])
    gt_vti = np.array([auc(spline[:,0], spline[:,1]) for spline in gt_splines])
    return pred_vti, gt_vti

def get_ejection_time_point(pred_kpts, gt_kpts, labels, err_metric):
    beg_index = next((index for index, label in enumerate(labels) if label  == 'ejection beginning'), -1)
    end_index = next((index for index, label in enumerate(labels) if label  == 'maximum velocity'), -1)
    pred_eje_time = np.apply_along_axis(ejection_time, 1, pred_kpts[:,:,0],beg_index, end_index)
    gt_eje_time  = np.apply_along_axis(ejection_time, 1,gt_kpts[:,:,0], beg_index, end_index)
    res = []
    for i in range(len(pred_kpts)):
        res.append(err_metric([gt_eje_time[i]],[pred_eje_time[i]]))
    return res

def get_maximum_velocity(pred_kpts, gt_kpts):
    pred_max_vel = np.amax(pred_kpts, axis=1)[:,1]
    gt_max_vel = np.amax(gt_kpts, axis=1)[:,1]
    return pred_max_vel, gt_max_vel

def get_metric(pred_kpts, gt_kpts, metric):
    res = []
    res.append(metric(gt_kpts.flatten(), pred_kpts.flatten()))
    for i in range(len(pred_kpts[0])):
        res.append(metric( gt_kpts[:,i], pred_kpts[:,i]))
    return res

def compute_mse(pred_kpts, gt_kpts):
    return mean_squared_error(gt_kpts, pred_kpts)

def compute_kpts_err(pred_kpts, gt_kpts):
    total = []
    for i in range(len(pred_kpts[0])):
        total.append(mean_squared_error(gt_kpts[:,i,:], pred_kpts[:,i,:], multioutput='raw_values', squared=False)) #returns array of shape [13, 2]
    total = np.array(total)
    all_points = np.mean(total, 1) 
    return np.mean(all_points)

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

def spline_point_count(labels):
    counter = 0 
    for i in range(len(labels)):
        if labels[i] == 'spline_point':
            labels[i] = labels[i] +'_'+ str(counter)
            counter += 1
    return labels 

def create_point_annotations_excel(pred_kpts, gt_kpts, phys_pred_kpts, phys_gt_kpts, labels, dict_keys, output_dir):
    columns= []
    labels = spline_point_count(labels)

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

def rmse_report(pred_kpts, gt_kpts, df, type = 'PIX'):
    #compute per point per axis
    total_per_dim = []
    total_per_point = []
    for i in range(len(pred_kpts[0])):
        total_per_dim.append(mean_squared_error(gt_kpts[:,i,:], pred_kpts[:,i,:], multioutput='raw_values', squared=False)) #returns array of shape [13, 2]
        total_per_point.append(mse_error_2dim(gt_kpts[:,i,:], pred_kpts[:,i,:]))
    total_per_dim = np.array(total_per_dim)
    
    total_x = mean_squared_error(gt_kpts[:,:,0].flatten(), pred_kpts[:,:,0].flatten(), squared=False)
    total_y = mean_squared_error(gt_kpts[:,:,1].flatten(), pred_kpts[:,:,1].flatten(), squared=False)
    total = mse_error_2dim(gt_kpts.reshape(-1,2), pred_kpts.reshape(-1,2))

    df.loc[type + ' RMSE X'] = np.insert(total_per_dim[:,0], 0, total_x) #shape [13 + 1, 1] 
    df.loc[type + ' RMSE Y'] = np.insert(total_per_dim[:,1], 0, total_y) #shape [13 + 1, 1] 
    
    df.loc[type + ' RMSE'] = np.insert(total_per_point, 0, total) #does mean of array all_points

    return df

def stats_report_doppler(pred_kpts, gt_kpts, phys_pred_kpts, phys_gt_kpts, gt_splines, pred_splines, labels):
    labels = spline_point_count(labels)
    pf = pd.DataFrame(columns= ['TOTAL'] + labels ,
                       index=[#"PIX MAPE", "PIX MAPE X", "PIX MAPE Y" 
                            "PIX RMSE", "PIX RMSE X", "PIX RMSE Y" 
                            #,"PHYS MAPE", "PHYS MAPE X", "PHYS MAPE Y"
                            ,"PHYS RMSE", "PHYS RMSE X","PHYS RMSE Y" 
                            ,"PUL IDX MAPE", "PUL IDX RMSE"
                            ,"EJE TIME MAPE", "EJE TIME RMSE"
                            ,"VTI MAPE", "VTI RMSE"
                            ,"MAX VEL MAPE", "MAX VEL RMSE"])

    pf = rmse_report(pred_kpts, gt_kpts, pf, type ='PIX')

    pf = rmse_report(phys_pred_kpts, phys_gt_kpts, pf, type ='PHYS')

    pf.loc[["MAX VEL MAPE"],["TOTAL"]] = mean_absolute_percentage_error(phys_gt_kpts[:,2,1],phys_pred_kpts[:,2,1])
    pf.loc[["MAX VEL RMSE"],["TOTAL"]] = mean_squared_error(phys_gt_kpts[:,2,1],phys_pred_kpts[:,2,1], squared=False )
     
    pred_pul_idx, gt_pul_idx = get_pulsatily_idx(pred_splines, gt_splines)
    pf.loc[["PUL IDX MAPE"],["TOTAL"]] = mean_absolute_percentage_error(gt_pul_idx,pred_pul_idx)
    pf.loc[["PUL IDX RMSE"],["TOTAL"]] = mean_squared_error(gt_pul_idx,pred_pul_idx, squared=False )
   
    pred_eje_time, gt_eje_time = get_ejection_time(phys_pred_kpts, phys_gt_kpts,labels)
    pf.loc[["EJE TIME MAPE"],["TOTAL"]] = mean_absolute_percentage_error(gt_eje_time, pred_eje_time)
    pf.loc[["EJE TIME RMSE"],["TOTAL"]] = mean_squared_error(gt_eje_time, pred_eje_time, squared=False)
    
    pred_vti, gt_vti = get_vti(pred_splines, gt_splines)
    pf.loc[["VTI MAPE"],["TOTAL"]] = mean_absolute_percentage_error(gt_vti, pred_vti)
    pf.loc[["VTI RMSE"],["TOTAL"]] = mean_squared_error(gt_vti, pred_vti, squared=False)
    
    pf = pf.astype(float)

    # idxs = np.argwhere(phys_gt_kpts == 0 )
    # print(idxs)

    return pf

def create_bland_altman(phys_pred_kpts, phys_gt_kpts,  gt_splines, pred_splines, labels, output_dir):
    with mpl.rc_context({'font.size': 25}):  
        #maximum velocity bland altman 
        plt.clf()
        ax = plt.gca()
        pred_max_vel, gt_max_vel = get_maximum_velocity(phys_pred_kpts, phys_gt_kpts)
        sm.graphics.mean_diff_plot(gt_max_vel, pred_max_vel, ax = ax, font_size = 30)
        ax.set_title("Maximum Velocity")
        plt.savefig(os.path.join(output_dir,'max_vel_BA.png'))

        plt.clf()
        ax = plt.gca()
        pred_pul_idx, gt_pul_idx = get_pulsatily_idx(pred_splines, gt_splines)
        sm.graphics.mean_diff_plot(gt_pul_idx, pred_pul_idx, ax = ax, font_size = 30)
        ax.set_title("Pulsatility Index")
        plt.savefig(os.path.join(output_dir,'pul_idx_BA.png'))

        plt.clf()
        ax = plt.gca()
        pred_eje_time, gt_eje_time = get_ejection_time(phys_pred_kpts, phys_gt_kpts,labels)
        sm.graphics.mean_diff_plot(gt_eje_time, pred_eje_time, ax = ax, font_size = 30)
        ax.set_title("Ejection time")
        plt.savefig(os.path.join(output_dir,'eje_time_BA.png'))

        plt.clf()
        ax = plt.gca()
        pred_vti, gt_vti = get_vti(pred_splines, gt_splines)
        sm.graphics.mean_diff_plot(gt_vti, pred_vti, ax = ax, font_size = 30)
        ax.set_title("Velocity time integral")
        plt.savefig(os.path.join(output_dir,'VTI_BA.png'))

def count_inversions(predKpts, img_paths, output_dir):
    inversions = 0
    text = ""
    for i, pred in enumerate(predKpts):
        if not np.all(np.diff(pred[:,0]) >= 0):
            inversions += 1
            text += "\nInversion in image {}".format(img_paths[i])
            text += np.array2string(pred[:,0])
    text = "\nTotal number of inverted images: {} \n\n ".format(inversions) + text
    with open(os.path.join(output_dir, "inversions.txt"), "w") as fp:
        fp.write(text)


