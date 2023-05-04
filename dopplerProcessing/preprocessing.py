from dirsParser import parse_arguments_directories
import cv2
import random
import numpy as np
from dopplerProcessing.utils import *
from dopplerProcessing.kpts_utils import *
from dopplerProcessing.visualization import draw_points_interactive, draw_img_interactive

def crop_image_cycles (pixel_array_rgb, cycle_metadata, margin=30):

    min_x, min_y, max_x, max_y = cycle_metadata["min_x"], cycle_metadata["min_y"], cycle_metadata["max_x"], cycle_metadata["max_y"]
    dims = cycle_metadata["dims"]

    crop_pixels = pixel_array_rgb[min_y - margin : max_y + margin
                                 ,min_x - margin : max_x + margin
                                 ,:]

    if dims is not None: #resize
        ratio = 1
        if crop_pixels.shape[0] > DIMS [0] or crop_pixels.shape[1]  > DIMS [1]:
            ratio = float(DIMS[1])/max(crop_pixels.shape[0],crop_pixels.shape[1])
            w = int(crop_pixels.shape[1] * ratio)
            h = int(crop_pixels.shape[0] * ratio)
            crop_pixels = cv2.resize(crop_pixels, (w,h), interpolation = cv2.INTER_AREA)

        t = max((dims[0] - crop_pixels.shape[0])//2, 0) 
        b = max(dims[0] - crop_pixels.shape[0] - t, 0) 
        l = max((dims[1] - crop_pixels.shape[1])//2, 0) 
        r = max(dims[1] - crop_pixels.shape[1] - l, 0)
        
        crop_pixels= cv2.copyMakeBorder(crop_pixels.copy() ,0,0, l//3, r//3,cv2.BORDER_REPLICATE)
        crop_pixels= cv2.copyMakeBorder(crop_pixels.copy() ,0,0, l - (l//3), r - (r//3),cv2.BORDER_REFLECT)
        crop_pixels= cv2.copyMakeBorder(crop_pixels.copy(),t, b, 0, 0,cv2.BORDER_REPLICATE)

        cycle_metadata["x_shift"] = l 
        cycle_metadata["y_shift"] = t
        cycle_metadata["margin"] = margin
        cycle_metadata["ratio"] = ratio

    return crop_pixels, cycle_metadata     

def process_image (img_file, output_file):
    ds = load_dicom(img_file)
    if ds is None: #controls if it is a dicom
        return 0
    if isSegmented(ds):
        return 0
    
    output_name_gen =  transform_name_no_dir (img_file) #get name 
    pixel_array_rgb = load_image(ds) #get pixel array

    #get kpts and splines
    json = get_json_image(img_file, json_data)
    patient = json["patient_id"]

    final_points_x, final_points_y, labels = get_annotations(json)
    interp_x, interp_y = get_splines(json)
    num_cycles = len(final_points_x)

    img_metadata = {}
    img_metadata["gen"] = get_img_metadata(ds) #to compute the physical metrics afterwars
    patient_files = []

    #iterate over cycles
    for i in range(num_cycles):
        cycle_metadata = {"min_x": min(final_points_x[i]),
                          "min_y": min(final_points_y[i]),
                          "max_x": max(final_points_x[i]),
                          "max_y": max(final_points_y[i]),
                          "dims": DIMS,
                          }
        output_name = output_name_gen + '_' + str(i)

        crop_pixels, cycle_metadata = crop_image_cycles(pixel_array_rgb, cycle_metadata)

        #transform point coordinates 
        cicle_points_x = transform_list(final_points_x[i], cycle_metadata, axis="x" ) 
        cicle_points_y = transform_list(final_points_y[i], cycle_metadata, axis="y" )
        cicle_interp_x = transform_list(interp_x[i], cycle_metadata, axis="x" )
        cicle_interp_y = transform_list(interp_y[i], cycle_metadata, axis="y" ) 

        #compute kpts 
        kpts_x, kpts_y, kpts_labels = compute_kpts(cicle_points_x, 
                                                   cicle_points_y,
                                                   cicle_interp_x, 
                                                   cicle_interp_y, 
                                                   labels[i], DESIRED_LABELS)
        
        kpts_saved = save_kpts(kpts_x, kpts_y, kpts_labels, output_dir, output_name, NUM_KPTS, DIMS)

        #save the image if everything went well 
        img_saved = cv2.imwrite(output_dir +"/frames/" + output_name + '.png', cv2.cvtColor(crop_pixels, cv2.COLOR_RGB2GRAY ))
        fig = draw_img_interactive(crop_pixels)
        draw_points_interactive(kpts_x, kpts_y, kpts_labels, fig)
        fig.write_html(output_dir +"/validations/" + output_name + '.html')

        #store the image metadata and name to the processed files file
        if kpts_saved and img_saved:
            patient_files.append(output_name)

        img_metadata[str(i)] = cycle_metadata
    np.save( output_dir + '/metadata/' + output_name_gen + '.npy', img_metadata)

    if patient in list_images.keys(): #check that we do not put same patient in test and train 
        list_images[patient] = list_images[patient] + patient_files
    else:
        list_images[patient] = patient_files

if __name__ == "__main__": 
    #dimensions
    DIMS = [300,300]
    TRAIN = 0.6
    TEST = 0.2
    VAL = 0.15
    TEST_AUX = 0.05

    assert TRAIN + TEST_AUX + TEST + VAL == 1, "The percentages should add up to one."

    DESIRED_LABELS = ['ejection beginning', 'mid upstroke', 'maximum velocity','mid deceleration point', 'ejection end', 'diastolic peak', 'ejection beginning']
    NUM_KPTS = len(DESIRED_LABELS)*2-1

    input_dir, output_dir , json_path = parse_arguments_directories(json=True)

    list_images = {} 

    #load json 
    json_data = load_json(json_path)

    #create directories
    create_dir(output_dir + '/frames')
    create_dir(output_dir + '/annotations')
    create_dir(output_dir + '/metadata')
    create_dir(output_dir + '/filenames')
    create_dir(output_dir + '/validations')

    _ = open_directory(input_dir, output_dir + '/frames', process_image)

    #distribute images
    keys = list_images.keys()
    random.shuffle(list(keys)) 
    list_images_shuffled = [list_images[key] for key in keys]
    train_files, test_files, val_files, test_aux_files = distribute_dataset(output_dir,"doppCycle",list_images_shuffled, TRAIN, TEST, VAL)

    number_patients_processed = len(list_images)
    number_images_created = len([img for patient in list_images for img in patient])

    with open(output_dir + "/count.txt", "w") as fp0:
        fp0.write("{} files have been processed.\n".format(number_patients_processed))
        fp0.write("{} images have been created.\n".format(number_images_created))
        fp0.write("The number of kpts is: {}.\n".format(NUM_KPTS))
        fp0.write("The dimenson is: {}.\n".format(DIMS))
        fp0.write("The train dataset has {} samples.\n".format(train_files))
        fp0.write("The test dataset has {} samples.\n".format(test_files))
        fp0.write("The validation dataset has {} samples.\n".format(val_files))
        fp0.write("The auxiliary test dataset has {} samples.\n".format(test_aux_files))

