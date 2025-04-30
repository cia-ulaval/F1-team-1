from curses import window
from itertools import count
from typing import final
from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils
import os
import json
import time
from os import walk

import numpy as np
from libemg.offline_metrics import OfflineMetrics

import socket

from drive import accelerate , brake , steer_left , steer_right , reset_controls
import vgamepad as vg 



WINDOW_SIZE = 200 # 40
WINDOW_INC = 20
CLASSES = [0, 1, 2 , 3, 4]
REPS = [0, 1, 2, 3, 4 , 5]
STAGE = 4 # 0: collect data, 1: prepare model, 2: test band, 3: test model, 4: prepare emg model, 5: prepare emg imu ppg model

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

    feature_dic_0 , windows_0 = get_training_data(folder_location="data/S" + str("0") + "/", regex_filters= emg_regex_filters)
    feature_dic_1 , windows_1 = get_training_data(folder_location="data/S" + str("1") + "/", regex_filters= emg_regex_filters)
    feature_dic_2 , windows_2 = get_training_data(folder_location="data/S" + str("2") + "/", regex_filters= emg_regex_filters)
    feature_dic_3 , windows_3 = get_training_data(folder_location="data/S" + str("3") + "/", regex_filters= emg_regex_filters)
    feature_dic_4 , windows_4 = get_training_data(folder_location="data/S" + str("4") + "/", regex_filters= emg_regex_filters)

    # for key , value in feature_dic_0.items():
    #     print("EMG features 0 : \n", key , value)

    windows = np.concatenate(( windows_0,windows_1,windows_2,windows_3,windows_4))

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

    # Concatenate MFL features
    feature_dic['training_features']["MFL"] = np.concatenate((
        feature_dic_0['training_features']['MFL'],
        feature_dic_1['training_features']['MFL'],
        feature_dic_2['training_features']['MFL'],
        feature_dic_3['training_features']['MFL'],
        feature_dic_4['training_features']['MFL']
    ))
    
    # Concatenate MSR features
    feature_dic['training_features']["MSR"] = np.concatenate((
        feature_dic_0['training_features']['MSR'],
        feature_dic_1['training_features']['MSR'],
        feature_dic_2['training_features']['MSR'],
        feature_dic_3['training_features']['MSR'],
        feature_dic_4['training_features']['MSR']
    ))

    # Concatenate WAMP features
    feature_dic['training_features']["WAMP"] = np.concatenate((
        feature_dic_0['training_features']['WAMP'],
        feature_dic_1['training_features']['WAMP'],
        feature_dic_2['training_features']['WAMP'],
        feature_dic_3['training_features']['WAMP'],
        feature_dic_4['training_features']['WAMP']
    ))
    
    # Concatenate all labels
    feature_dic['training_labels'] = np.concatenate((
        feature_dic_0['training_labels'],
        feature_dic_1['training_labels'],
        feature_dic_2['training_labels'],
        feature_dic_3['training_labels'],
        feature_dic_4['training_labels']
    ))


    model = emg_predictor.EMGClassifier("LDA")
    model.fit(feature_dictionary=feature_dic)
    model.add_velocity(windows, feature_dic['training_labels'])
    
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', 
                                                    ecg=False, 
                                                    imu=False, 
                                                    ppg=False, 
                                                    eda=False, 
                                                    emg=True,
                                                    filtering=True,
                                                    emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    feature_list = feature_extractor.FeatureExtractor().get_feature_groups()['LS4']

    oc = emg_predictor.OnlineEMGClassifier(model, WINDOW_SIZE, WINDOW_INC, odh, feature_list, std_out=False)
    import socket
    UDP_IP = "127.0.0.1"
    UDP_PORT = 12346
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    # Démarrer le classificateur dans un thread
    import threading
    classifier_thread = threading.Thread(target=oc.run, args=(True,))
    classifier_thread.start()

    print(f"En attente de prédictions sur {UDP_IP}:{UDP_PORT}...")

    while True:
        data, _ = sock.recvfrom(10)
        message = data.decode("utf-8").strip()
        print(f"Message reçu brute: {message}")
        try:
            prediction_str, velocity_str = message.split(" ")
            prediction = int(prediction_str)
            velocity = float(velocity_str)
            print(f"Commande reçue: Classe={prediction}, Vitesse={velocity}")
            #"class_map": {"0": "Hand_Open", "1": "Thumbs_Flexion", "2": "Thumbs_Up", "3": "Wrist_Left_Rotation", "4": "Wrist_Right_Rotation"}
            if prediction == 0:
                reset_controls()
            # if prediction == 1:
            #     accelerate()
            # if prediction == 2:
            #     brake()
            # if prediction == 3:
            #     steer_left()
            # if prediction == 4:
            #     steer_right()

        except Exception as e:
            print(f"Erreur: {e}")

    


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

    odh = fi_emg.filter(odh)

    fe = feature_extractor.FeatureExtractor()
    # LS4 feature set for EMG
    feature_dic = {}
    emg_features = fe.extract_feature_group("LS4", windows)
    
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
        preparemodel()
    if STAGE == 2:
        testband()
    if STAGE == 3:
        testmodel()
    if STAGE == 4:
        prepareemgmodel()
    if STAGE == 5:
        prepareemgimuppgmodel()