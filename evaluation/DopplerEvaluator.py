import numpy as np
import copy
import cv2
import random
import logging
import os
import shutil
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pickle
import pandas as pd 

from datasets import datas
from .BaseEvaluator import DatasetEvaluator
from dopplerProcessing.visualization import  plot_kpts_pred_and_gt

from dopplerProcessing.kpts_utils import to_physical_list
from dopplerProcessing.stat_utils import stats_report_doppler, compute_kpts_err, stats_report_point_doppler, create_bland_altman

class DopplerEvaluator(DatasetEvaluator):
    """
    Evaluate EchoNet segmentation predictions for a single iteration of the cardiac navigation model
    """

    def __init__(
        self,
        dataset: datas,
        output_dir: str = "./visu",
        verbose: bool = True
    ):
        """
        Args:
            dataset (dataset object): Note: used to be dataset_name: name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "single_iter", "multi_iter".
                By default, will infer this automatically from predictions.
            output_dir (str): optional, an output directory to dump results.
        """
        self._tasks = ["kpts"]
        self._dataset = dataset
        self._verbose = verbose
        self._output_dir = output_dir
        if self._verbose:
            self.set_logger(logname=os.path.join(output_dir, "eval_log.log"))
            self._logger = logging.getLogger(__name__)

        self._cpu_device = torch.device("cpu")
        self._do_evaluation = True 

    def reset(self):
        self._predictions = dict()

    def set_logger(self, logname):
        print("Evaluation log file is set to {}".format(logname))
        logging.basicConfig(filename=logname,
                            filemode='w', #'a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)    #level=logging.DEBUG)    # level=logging.INFO)


    def process(self, inputs: Dict, outputs: Dict) -> None:
        """
        Args:
            inputs: the inputs to a EF and Kpts model. It is a list of dicts. Each dict corresponds to an image and
                contains keys like "keypoints", "ef".
            outputs: the outputs of a EF and Kpts model. It is a list of dicts with keys
                such as "ef_prediction" or "keypoints_prediction" that contains the proposed ef measure or keypoints coordinates.
        """
        some_val_output_item = next(iter(outputs.items()))[1]

        self._predictions = dict()
        for ii, data_path in enumerate(outputs):
            prediction = dict()

            # get predictions:
            if some_val_output_item["keypoints_prediction"] is not None:
                prediction["keypoints_prediction"] = outputs[data_path]["keypoints_prediction"]
                prediction["keypoints"] = inputs[data_path]["keypoints"]

            # get case name:
            prediction["data_path_from_root"] = data_path.replace(self._dataset.img_folder, "")

            self._predictions[data_path] = prediction


    def evaluate(self, tasks: List = None):
        if tasks is not None:
            self._tasks = tasks

        predictions = self._predictions

        if len(predictions) == 0 and self._verbose:
            self._logger.warning("[DopplerEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir is not None:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "echonet_predictions.pkl")
            with open(file_path, 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not self._do_evaluation and self._verbose:
            self._logger.info("Annotations are not available for evaluation.")
            return

        if self._verbose:
            self._logger.info("Evaluating predictions ...")
        self._results = OrderedDict()
        res = self._eval_keypoints_predictions(predictions, report = self._output_dir is not None )
        self._results["kpts"] = res

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def plot(self, num_examples_to_plot: int) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(16, 16))
        plot_directory = os.path.join(self._output_dir, "plots")
        if os.path.exists(plot_directory):
            shutil.rmtree(plot_directory)
        os.makedirs(plot_directory)
        self._logger.info("plotting {} prediction examples to {}".format(num_examples_to_plot, plot_directory))
        for data_path in random.sample(list(self._predictions), num_examples_to_plot):
            prediction = self._predictions[data_path]
            fig.clf()
            fig = self._plot_kpts_single_frame(fig, data_path_from_root=prediction["data_path_from_root"],
                                                keypoints_prediction=prediction["keypoints_prediction"])
            plot_filename = "{}.jpg".format(os.path.splitext(prediction["data_path_from_root"])[0].replace("/", "_"))
            fig.savefig(fname=os.path.join(plot_directory, plot_filename))

    def set_tasks(self, tasks: List) -> None:
        self._tasks = tasks

    def get_tasks(self) -> List:
        return self._tasks

    def _eval_keypoints_predictions(self, predictions: Dict, report = False) -> Dict:
        """
        Evaluate keypoints predictions
        Args:
            predictions (list[dict]): list of predictions from the model
        """
        if self._verbose:
            self._logger.info("Eval stats for keypoints")


        all_pred_kpts = []
        all_phys_pred_kpts = []
        all_gt_kpts = []
        all_phys_gt_kpts = []
        img_paths = []

        for prediction in predictions.values():
            datapoint_index = self._dataset.img_list.index(prediction["data_path_from_root"])
            metadata,cycle = self._dataset.get_metadata(datapoint_index)

            #denormalize keypoints from 0-1 to 0-size of image
            gt_kpts = prediction["keypoints"]  * self._dataset.input_size
            pred_kpts = prediction["keypoints_prediction"] * self._dataset.input_size

            #convert to physical keypoints
            if report:
                phys_gt_kpts = to_physical_list(gt_kpts, metadata, cycle)
                phys_pred_kpts = to_physical_list(pred_kpts, metadata, cycle)
                all_phys_gt_kpts.append(phys_gt_kpts)
                all_phys_pred_kpts.append(phys_pred_kpts)
                img_paths.append(prediction["data_path_from_root"])

            all_gt_kpts.append(gt_kpts)
            all_pred_kpts.append(pred_kpts)

        all_gt_kpts = np.array(all_gt_kpts)
        all_pred_kpts = np.array(all_pred_kpts)

        if report:
            all_phys_gt_kpts = np.array(all_phys_gt_kpts)
            all_phys_pred_kpts = np.array(all_phys_pred_kpts)
            labels = self._dataset.get_labels(datapoint_index)

            df = stats_report_doppler(all_pred_kpts,all_gt_kpts, all_phys_pred_kpts, all_phys_gt_kpts,labels)
            df2 = stats_report_point_doppler(all_pred_kpts,all_gt_kpts, all_phys_pred_kpts, all_phys_gt_kpts,labels,img_paths)
            
            create_bland_altman(all_pred_kpts,all_gt_kpts, all_phys_pred_kpts, all_phys_gt_kpts,labels, self._output_dir)
            
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            file_path = os.path.join(self._output_dir, "stats_report.csv")
            df.to_csv(file_path, sep=";", decimal=",")

            file_path = os.path.join(self._output_dir, "stats_point_report.csv")
            df2.to_csv(file_path, sep=";", decimal=",")

            mKptsERR = df.loc["PIX MSE"]["TOTAL"]

            if self._output_dir is not None:
                if not os.path.exists(self._output_dir):
                    os.makedirs(self._output_dir)
                file_path = os.path.join(self._output_dir, "echonet_predictions_2.pkl")
                with open(file_path, 'wb') as handle:
                    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        else:
            mKptsERR = compute_kpts_err(all_pred_kpts, all_gt_kpts)

        if self._verbose:
            self._logger.info("Mean keypoints error is {}".format(mKptsERR))
        return mKptsERR


    def _plot_kpts_single_frame(self, fig, data_path_from_root, keypoints_prediction):
        datapoint_index = self._dataset.img_list.index(data_path_from_root)
        data = self._dataset.get_img_and_kpts(datapoint_index)
        img = data["img"]
        keypoints = data["kpts"]
        # normalize:
        keypoints = self._dataset.normalize_pose(keypoints, img)
        img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        keypoints_prediction = self._dataset.denormalize_pose(keypoints_prediction, img)
        keypoints = self._dataset.denormalize_pose(keypoints, img)

        plot_kpts_pred_and_gt(fig, img, gt_kpts=keypoints, pred_kpts=keypoints_prediction,
                              kpts_info=self._dataset.kpts_info, closed_contour=self._dataset.kpts_info['closed_contour'])

        #prediction_text = "Keypoints err: {:.2f}".format(np.mean(dist_pred_gt_kpts[img_index]))
        return fig
