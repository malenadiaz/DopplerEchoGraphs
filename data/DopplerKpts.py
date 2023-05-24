import torch
import os
import numpy as np
import cv2
import albumentations as A
from PIL import Image
import glob
from typing import List, Dict
import matplotlib.pyplot as plt

# internal:
from data.USKpts import USKpts
from utils.utils_plot import plot_grid, draw_kpts, plot_kpts_pred_and_gt

class DopplerKpts(USKpts):
    def __init__(self, dataset_config, filenames_list: str = None, transform: A.core.composition.Compose = None):
        self.LABELS = []

        super().__init__(dataset_config, filenames_list, transform)
        self.metadata_dir = self.img_folder.replace("frames", "metadata")
        self.COLORS = [(0, 0, 255), (255, 127, 0), (255, 70, 0), (0, 174, 174), (186, 0, 255), 
          (255, 255, 0), (204, 0, 175), (255, 0, 0), (0, 255, 0), (115, 8, 165), 
          (254, 179, 0), (0, 121, 0), (0, 0, 255)]
        
    def create_img_list(self, filenames_list: str) -> None:
        """
        Called during construction. Creates a list containing paths to frames in the dataset
        """
        img_list_from_file = []
        with open(filenames_list) as f:
            img_list_from_file.extend(f.read().splitlines())
        self.img_list = img_list_from_file
    
    def img_to_torch(self, img: np.ndarray) -> torch.Tensor:
        """ Convert original image format to torch.Tensor """
        # resize:
        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR_EXACT)
        # transform:
        img = Image.fromarray(np.uint8(img))
        img = self.basic_transform(img)

        return img
    
    def load_kpts_annotations(self, img_list: List) -> np.ndarray:
        """ Creates an array of annotated keypoints coorinates for the frames in the dataset. """
        KP_COORDS = []
        if self.anno_dir is not None:
            for fname in img_list:
                kpts = np.load(os.path.join(self.anno_dir, fname.replace("png", "npy")), allow_pickle=True)
                KP_COORDS.append(kpts[:,0:2])
                if len(self.LABELS) == 0 :
                    self.LABELS.append(kpts[:,2])
            #KP_COORDS = np.array(KP_COORDS).swapaxes(0, 1)

        return KP_COORDS

    def get_img_and_kpts(self, index: int):
        """
        Load and parse a single data point.
        Args:
            index (int): Index
        Returns:
            img (ndarray): RGB frame in required input_size
            kpts (ndarray): Denormalized, namely in img coordinates
            img_path (string): full path to frame file in image format (PNG or equivalent)
        """
        # ge paths:
        img_path = os.path.join(self.img_folder, self.img_list[index])
        # get image: (PRE-PROCESS UNIQUE TO UltraSound data)
        img = cv2.imread(img_path)
        kpts = np.zeros([self.num_kpts, 2])     # default
        if self.anno_dir is not None:
            #pos = [i for i in range(5)] +[6] + [i for i in range(8,13)] #select 11pts
            #pos = [i for i in range(0,13,2)] #select even points
            kpts = self.KP_COORDS[index].astype(int)


        # resize to DNN input size:
        ratio = [self.input_size / float(img.shape[1]), self.input_size / float(img.shape[0])]
        if img.shape[0] != self.input_size:
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR_EXACT)
        # resizing keypoints:
        kpts = np.round(kpts * ratio)   #also cast int to float

        data = {"img": img,
                "kpts": kpts,
                "img_path": img_path
                # "ignore_margin": ignore_margin,
                # "ratio": ratio
                }
        return data

    def get_metadata(self, index:int):
        if self.metadata_dir is not None:
            og_name = self.img_list[index]
            name_list = og_name.split("_")
            filename = "_".join(name_list[:-1])
            cycle = name_list[-1][0]
            metData = np.load(os.path.join(self.metadata_dir, filename +  ".npy"), allow_pickle=True)
        else:
            metData = None
        return metData.item(), cycle
    
    def get_labels(self):
        #pos = [i for i in range(0,13,2)]
        return self.LABELS[0]
    
    def get_colors(self):
        return self.COLORS