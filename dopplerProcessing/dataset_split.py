import itertools
import random
import os 
from dirsParser import parse_arguments_directories
from utils import distribute_dataset

def split_dataset(input_dir):
    files = os.listdir(input_dir + "/frames")

    files = list(map(lambda x: (x, x[:-6]), files))
    results = []
    for key, group in itertools.groupby(sorted(files), key=lambda x: x[1]):
        client_files = [x[0] for x in group]
        results.append(client_files)

    return results


if __name__ == "__main__": 
    TRAIN = 0.6
    TEST = 0.2
    VAL = 0.1
    TEST_AUX = 0.1

    assert TRAIN + TEST + VAL + TEST_AUX == 1, "The percentages should add up to one."

    input_dir, output_dir , json_path = parse_arguments_directories()

    list_images = split_dataset(input_dir)

    random.shuffle(list_images)
    number_files_created = len(list_images)


    random.shuffle(list_images) 
    train_files, test_files, val_files, test_aux_files = distribute_dataset(output_dir,"doppCycle",list_images, TRAIN, TEST, VAL)

    number_patients_processed = len(list_images)
    number_images_created = len([img for patient in list_images for img in patient])

    with open(output_dir + "/count.txt", "w") as fp0:
        fp0.write("{} files have been processed.\n".format(number_patients_processed))
        fp0.write("{} images have been created.\n".format(number_images_created))
        fp0.write("The train dataset has {} samples.\n".format(train_files))
        fp0.write("The test dataset has {} samples.\n".format(test_files))
        fp0.write("The validation dataset has {} samples.\n".format(val_files))
        fp0.write("The auxiliary test dataset has {} samples.\n".format(test_aux_files))


    
