from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils
import os
import json
import time
from os import walk


WINDOW_SIZE = 200 # 40
WINDOW_INC = 20
CLASSES = [0, 1, 2 , 3, 4]
REPS = [0, 1, 2, 3, 4 , 5]
STAGE = 0

def testband():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3',  ecg=True,  imu=True, ppg=True, eda=True, emg=True,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    #odh is your data
    odh.visualize(num_samples=10000)

def preparemodel():
    regex_filters = [

    data_handler.RegexFilter(left_bound = "C_", right_bound="_R_", values = [str(i) for i in REPS], description='reps'),
    data_handler.RegexFilter(left_bound = "_R_", right_bound="_emg.csv", values = [str(i) for i in CLASSES], description='classes'),

    data_handler.RegexFilter(left_bound = "C_", right_bound="_R_", values = [str(i) for i in REPS], description='reps'),
    data_handler.RegexFilter(left_bound = "_R_", right_bound="_imu.csv", values = [str(i) for i in CLASSES], description='classes'),

    data_handler.RegexFilter(left_bound = "C_", right_bound="_R_", values = [str(i) for i in REPS], description='reps'),
    data_handler.RegexFilter(left_bound = "_R_", right_bound="_ppg.csv", values = [str(i) for i in CLASSES], description='classes')
]
    """
    odh = data_handler.OfflineDataHandler()
    odh.get_data(folder_location="data/S" + str(0) + "/", regex_filters= regex_filters)"
    """

    odh = data_handler.OfflineDataHandler()
    for i in range(0, 7):
        odh.get_data(folder_location="data/S" + str(i) + "/", regex_filters= regex_filters)
    windows, metadata = odh.parse_windows(WINDOW_SIZE,WINDOW_INC)

    fe = feature_extractor.FeatureExtractor()
    feature_dic = {}
    feature_dic['training_features'] = fe.extract_feature_group("HTD", windows)
    feature_dic['training_labels'] = metadata['classes']
    model = emg_predictor.EMGClassifier("")
    model.fit(feature_dictionary=feature_dic)
    model.add_velocity(windows, metadata['classes'])

    return model

"""
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', ecg=False, imu=True, ppg=False, eda=False, emg=False,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    feature_list = feature_extractor.FeatureExtractor().get_feature_groups()['HTD']
    oc = emg_predictor.OnlineEMGClassifier(model, WINDOW_SIZE, WINDOW_INC, odh, feature_list, std_out=True)
    oc.run(True)"
"""

def testmodel():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', ecg=True, imu=True, ppg=True, eda=True, emg=True,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    feature_list = feature_extractor.FeatureExtractor().get_feature_groups()['HTD']
    oc = emg_predictor.OnlineEMGClassifier(preparemodel(), WINDOW_SIZE, WINDOW_INC, odh, feature_list, std_out=True)
    oc.run(True)
    

class CustomGui(gui.GUI):
    def __init__(self, 
                 online_data_handler,
                 args={'media_folder': 'images/', 'data_folder':'data/', 'num_reps': 3, 'rep_time': 5, 'rest_time': 3, 'auto_advance': True},
                 width=1920,
                 height=1080,
                 debug=False,
                 gesture_width=500,
                 gesture_height=500,
                 clean_up_on_kill=False):
        super().__init__(
            online_data_handler=online_data_handler,
            args=args,
            width=width,
            height=height,
            debug=debug,
            gesture_width=gesture_width,
            gesture_height=gesture_height,
            clean_up_on_kill=clean_up_on_kill
        )
    
    def download_gestures(self, gesture_ids, folder, download_imgs=True, download_gifs=False, redownload=False):
        """Downloads gesture images from repository"""
        import requests
        from pathlib import Path

        # Define repository paths
        git_url = "https://raw.githubusercontent.com/cia-ulaval/F1-team-1/main/f1/"
        gesture_data = {
            "1": "Hand_Open",
            "2": "Hand_Close",
            "3": "Wrist_Flexion",
            "4": "Wrist_Extension",
            "5": "Wrist_Pronation"
        }

        # Create output directory if it doesn't exist
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Download images
        for id in gesture_ids:
            try:
                idx = str(id)
                if idx not in gesture_data:
                    print(f"Warning: Gesture ID {id} not found in gesture list")
                    continue

                img_file = gesture_data[idx] + ".jpg"
                img_path = folder_path / img_file
                img_url = git_url + "images/" + img_file

                if download_imgs and (not img_path.exists() or redownload):
                    print(f"Downloading gesture {id}: {img_file}")
                    response = requests.get(img_url)
                    
                    if response.status_code == 200:
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Successfully downloaded {img_file}")
                    else:
                        print(f"Failed to download {img_file}: HTTP {response.status_code}")

            except Exception as e:
                print(f"Error downloading gesture {id}: {str(e)}")
                continue

def collectdata():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', ecg=True, imu=True, ppg=True, eda=True, emg=True,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    #odh is your data
    # odh.visualize_channels(list(range(8)), num_samples=10000)

    
    args = {
        "media_folder": "images/",
        "data_folder": "data/S" + str("4") + "/",
        "num_reps": 6,
        "rep_time": 5,
        "rest_time": 3,
        "auto_advance": True
    }
    guii = CustomGui(odh, args=args, debug=False)
    guii.download_gestures([1,2,3,4,5], "images/")
    guii.start_gui()
    

if __name__ == "__main__":
    if STAGE == 0:
        collectdata()
    if STAGE == 1:
        preparemodel()
    if STAGE == 2:
        testband()
    if STAGE == 3:
        testmodel()