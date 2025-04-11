from curses import window
from typing import final
from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils
import os
import json
import time
from os import walk

import numpy as np
from libemg.offline_metrics import OfflineMetrics


WINDOW_SIZE = 200 # 40
WINDOW_INC = 20
CLASSES = [0, 1, 2 , 3, 4]
REPS = [0, 1, 2, 3, 4 , 5]
STAGE = 1

def testband():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3',  ecg=True,  imu=True, ppg=True, eda=True, emg=True,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    #odh is your data
    odh.visualize(num_samples=10000)

def prepareemgmodel():

    emg_regex_filters = [
        data_handler.RegexFilter(left_bound = "C_", right_bound="_R_", values = [str(i) for i in REPS], description='reps'),
        data_handler.RegexFilter(left_bound = "_R_", right_bound="_emg.csv", values = [str(i) for i in CLASSES], description='classes')

    ]

    

    # for i in range(0, 5):
    #     odh_emg.get_data(folder_location="data/S" + str(i) + "/", regex_filters= emg_regex_filters)
    #     emg_windows, metadata = odh_emg.parse_windows(WINDOW_SIZE,WINDOW_INC)
    # print("emg_windows: ", emg_windows)
    # print("metadata: ", metadata)
        
    
    #Pre-processing
    #a standardization filter(EMG , IMU , PPG)
    #EMG data, application of a 60Hz notch filter + 20-450Hz band-pass filter
    
    
    #print("EMG features: ", emg_features)
    feature_dic_0 , windows_0  = get_training_data(folder_location="data/S" + str("0") + "/", regex_filters= emg_regex_filters)
    feature_dic_1 , windows_1 = get_training_data(folder_location="data/S" + str("1") + "/", regex_filters= emg_regex_filters)
    feature_dic_2 , windows_2 = get_training_data(folder_location="data/S" + str("2") + "/", regex_filters= emg_regex_filters)
    feature_dic_3 , windows_3 = get_training_data(folder_location="data/S" + str("3") + "/", regex_filters= emg_regex_filters)
    feature_dic_4 , windows_4 = get_training_data(folder_location="data/S" + str("4") + "/", regex_filters= emg_regex_filters)

    print("windows_0 shape: \n", windows_0.shape)
    print("windows_1 shape: \n", windows_1.shape)
    print("windows_2 shape: \n", windows_2.shape)
    print("windows_3 shape: \n", windows_3.shape)
    print("windows_4 shape: \n", windows_4.shape)

    windows = np.concatenate((
        windows_0,
        windows_1,
        windows_2,
        windows_3,
        windows_4
    ))
    print("windows shape: \n", windows.shape)

    # print("LS feature for feature_dic_0: \n", type(feature_dic_0["training_features"]["LS"]))
    # print("Training labels for feat 0: \n", feature_dic_0["training_labels"].shape)
    # print("LS feature for feature_dic_1: \n", feature_dic_1["training_features"]["LS"].shape)
    # print("Training labels for feat 1: \n", feature_dic_1["training_labels"].shape)
    # print("LS feature for feature_dic_2: \n", feature_dic_2["training_features"]["LS"].shape)
    # print("LS feature for feature_dic_3: \n", feature_dic_3["training_features"]["LS"].shape)
    # print("LS feature for feature_dic_4: \n", feature_dic_4["training_features"]["LS"].shape)
    # print("MFL feature for feature_dic_0: \n", feature_dic_0["training_features"]["MFL"].shape)
    # print("MFL feature for feature_dic_1: \n", feature_dic_1["training_features"]["MFL"].shape)
    # print("MFL feature for feature_dic_2: \n", feature_dic_2["training_features"]["MFL"].shape)
    # print("MFL feature for feature_dic_3: \n", feature_dic_3["training_features"]["MFL"].shape)
    # print("MFL feature for feature_dic_4: \n", feature_dic_4["training_features"]["MFL"].shape)
    # print("MSR feature for feature_dic_0: \n", feature_dic_0["training_features"]["MSR"].shape)
    # print("MSR feature for feature_dic_1: \n", feature_dic_1["training_features"]["MSR"].shape)
    # print("MSR feature for feature_dic_2: \n", feature_dic_2["training_features"]["MSR"].shape)
    # print("MSR feature for feature_dic_3: \n", feature_dic_3["training_features"]["MSR"].shape)
    # print("MSR feature for feature_dic_4: \n", feature_dic_4["training_features"]["MSR"].shape)
    # # print("feature_dic_1: ", feature_dic_1)
    # print("feature_dic_2: ", feature_dic_2)
    # print("feature_dic_3: ", feature_dic_3)
    # print("feature_dic_4: ", feature_dic_4)


    feature_dic = {
        'training_features': {},
        'training_labels': None
    }
    feature_dic['training_features']["LS"] = np.concatenate((
        feature_dic_0['training_features']['LS'],
        feature_dic_1['training_features']['LS'],
        feature_dic_2['training_features']['LS'],
        feature_dic_3['training_features']['LS'],
        feature_dic_4['training_features']['LS']
    ))

    #print("LS feature for final feature_dic: \n", feature_dic["training_features"]["LS"].shape)
    # Concatenate MFL features
    feature_dic['training_features']["MFL"] = np.concatenate((
        feature_dic_0['training_features']['MFL'],
        feature_dic_1['training_features']['MFL'],
        feature_dic_2['training_features']['MFL'],
        feature_dic_3['training_features']['MFL'],
        feature_dic_4['training_features']['MFL']
    ))
    #print("MFL feature for final feature_dic: \n", feature_dic["training_features"]["MFL"].shape)
    # Concatenate MSR features
    feature_dic['training_features']["MSR"] = np.concatenate((
        feature_dic_0['training_features']['MSR'],
        feature_dic_1['training_features']['MSR'],
        feature_dic_2['training_features']['MSR'],
        feature_dic_3['training_features']['MSR'],
        feature_dic_4['training_features']['MSR']
    ))
    #print("MSR feature for final feature_dic: \n", feature_dic["training_features"]["MSR"].shape)
    # Concatenate all labels
    feature_dic['training_labels'] = np.concatenate((
        feature_dic_0['training_labels'],
        feature_dic_1['training_labels'],
        feature_dic_2['training_labels'],
        feature_dic_3['training_labels'],
        feature_dic_4['training_labels']
    ))
    #print("All labels for final feature_dic: \n", feature_dic["training_labels"].shape)
    
    # print("EMG features: ", feature_dic['training_features'])
    # feature_dic['training_labels'] = np.hstack([feature_dic_0['training_labels'], feature_dic_1['training_labels'], feature_dic_2['training_labels'], feature_dic_3['training_labels'], feature_dic_4['training_labels']])
    # print("EMG labels: ", feature_dic['training_labels'])
    


    model = emg_predictor.EMGClassifier("LDA")
    model.fit(feature_dictionary=feature_dic)
    model.add_velocity(windows, feature_dic['training_labels'])


"""
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', ecg=False, imu=True, ppg=False, eda=False, emg=False,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    feature_list = feature_extractor.FeatureExtractor().get_feature_groups()['HTD']
    oc = emg_predictor.OnlineEMGClassifier(model, WINDOW_SIZE, WINDOW_INC, odh, feature_list, std_out=True)
    oc.run(True)"
"""

def get_training_data(folder_location, regex_filters):
    odh = data_handler.OfflineDataHandler()
    odh.get_data(folder_location=folder_location, regex_filters=regex_filters)
    windows, metadata = odh.parse_windows(WINDOW_SIZE,WINDOW_INC)
    fi_emg = filtering.Filter(2000)
    
    standardization_filter = { "name": "standardization" , }
    emg_notch_filter = { "name": "notch", "cutoff": 60, "bandwidth": 3}
    emg_bandpass_filter = { "name":"bandpass", "cutoff": [20, 450], "order": 4}

    fi_emg.install_filters(standardization_filter)
    fi_emg.install_filters(emg_notch_filter)
    fi_emg.install_filters(emg_bandpass_filter)

    fe = feature_extractor.FeatureExtractor()
    # LS4 feature set for EMG
    feature_dic = {}
    emg_features = fe.extract_feature_group("LS4", windows)
    #print("EMG features: ", emg_features)
    # print("EMG feature LS4 shapes: ", emg_features['LS'].shape)
    # print("EMG feature MFL shapes: ", emg_features['MFL'].shape)
    # print("EMG features MSR: ", emg_features['MSR'].shape)

    
    feature_dic['training_features'] = emg_features
    feature_dic['training_labels'] = metadata['classes']
    return feature_dic , windows

def prepareemgimuppgmodel():
    imu_regex_filters = [
        data_handler.RegexFilter(left_bound = "C_", right_bound="_R_", values = [str(i) for i in REPS], description='reps'),
        data_handler.RegexFilter(left_bound = "_R_", right_bound="_imu.csv", values = [str(i) for i in CLASSES], description='classes')
    ]

    ppg_regex_filters = [
        data_handler.RegexFilter(left_bound = "C_", right_bound="_R_", values = [str(i) for i in REPS], description='reps'),
        data_handler.RegexFilter(left_bound = "_R_", right_bound="_ppg.csv", values = [str(i) for i in CLASSES], description='classes')
    ]

    

    odh_imu = data_handler.OfflineDataHandler()
    for i in range(0, 5):
        odh_imu.get_data(folder_location="data/S" + str(i) + "/", regex_filters= imu_regex_filters)
        
    imu_windows, metadata = odh_imu.parse_windows(WINDOW_SIZE,WINDOW_INC)

    odh_ppg = data_handler.OfflineDataHandler()
    for i in range(0, 5):
        odh_ppg.get_data(folder_location="data/S" + str(i) + "/", regex_filters= ppg_regex_filters)
        
    ppg_windows, metadata = odh_ppg.parse_windows(WINDOW_SIZE,WINDOW_INC)

    #Pre-processing
    #a standardization filter(EMG , IMU , PPG)
    #EMG data, application of a 60Hz notch filter + 20-450Hz band-pass filter
    #PPG data, application of a 0.66-3Hz band-pass filter +  a Principal Component Analysis (PCA) is applied to the Infrared and Red channels
    #IMU data, the approach uses the derivative of the 3-axis accelerometer data from the standardized signals

    # fi_emg = filtering.Filter(2000)
    
    # standardization_filter = { "name": "standardization"}
    # emg_notch_filter = { "name": "notch", "cutoff": 60}
    # emg_bandpass_filter = { "name": "bandpass", "cutoff": [20, 450]}

    # fi_emg.install_filters(standardization_filter)
    # fi_emg.install_filters(emg_notch_filter)
    # fi_emg.install_filters(emg_bandpass_filter)

    # fi_ppg = filtering.Filter(100)
    # ppg_bandpass_filter = { "name": "bandpass", "cutoff": [0.66, 3]}
    # pca_filter = { "name": "pca", "num_components": 2}
    # fi_ppg.install_filters(standardization_filter)
    # fi_ppg.install_filters(ppg_bandpass_filter)

    # fi_imu = filtering.Filter(100)

    fe = feature_extractor.FeatureExtractor()
    imu_features = fe.extract_features(["WLPHASOR" , "DFTR" , "WENG" ,"RMSPHASOR"], imu_windows)
    #print("IMU features: ", imu_features)
    print("IMU feature WLPHASOR shapes: ", imu_features['WLPHASOR'].shape)
    print("IMU feature DFTR shapes: ", imu_features['DFTR'].shape)
    print("IMU feature WENG shapes: ", imu_features['WENG'].shape)
    print("IMU feature RMSPHASOR shapes: ", imu_features['RMSPHASOR'].shape)
    ppg_features = fe.extract_features(["MPK", "WENG", "MEAN"], ppg_windows)
    #print("PPG features: ", ppg_features)
    print("PPG feature MPK shapes: ", ppg_features['MPK'].shape)
    print("PPG feature WENG shapes: ", ppg_features['WENG'].shape)
    print("PPG feature MEAN shapes: ", ppg_features['MEAN'].shape)
    #print("PPG features: ", ppg_features)

    import numpy as np
    combined_features = np.hstack([
        emg_features['features'],  # Extract 'features' from each dictionary
        imu_features['features'],
        ppg_features['features']
    ])


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
        prepareemgmodel()
    if STAGE == 2:
        testband()
    if STAGE == 3:
        testmodel()