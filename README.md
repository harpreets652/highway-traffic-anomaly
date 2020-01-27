# Highway Traffic Anomaly(HTA) Data Set
The Highway Traffic Anomaly data set and related code.

## Setup
#### Prerequisites
* Python 3.7
* opencv-python
* opencv-contrib-python
* sk-video
* numpy

The HTA data set is composed of videos from the [Berkeley DeepDrive](https://bdd-data.berkeley.edu/) data set. Video
filenames are listed in the `data_set` directory in text files. The following steps explain how to construct the HTA 
data set: 
* Download the Berkeley DeepDrive data set
* Run **construct_dataset.py** for training set, and pos/neg test set
    * This script will search the DeepDrive directory for train/test set videos and copy them to a specified directory
        ```sh
      python construct_dataset.py --source_dir path_to_deep_drive_root --query_list data_set/training_set.txt --destination destination_path
* Vehicle accident anomaliees were downloaded from YouTube and aren't part of DeepDrive
    * Vehicle accident anomalies are available at [this](https://drive.google.com/open?id=1Y0lfKZ4owGEZKp48yKhN2XAppxGjZYiW) Google Drive 

Anomalous frame annotations are also in the `data_set` directory, anomalous-frames.txt

## Additional Files
* `utils` directory contains other helpful scripts.
    * `fix_orientation.py`: Some DeepDrive videos are rotated clockwise/counter clockwise. This script can fix that and save a new video
    * `motion_viz.py`: Script to visualize optical flow
* `models`: Deep Learning models, in Tensorflow/Keras, that were tested in this study
    * Requires Tensorflow, Keras 