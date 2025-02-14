from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils

WINDOW_SIZE = 200 # 40
WINDOW_INC = 20
CLASSES = [0, 1, 2, 3, 4]
REPS = [0, 1, 2, 3, 4]
STAGE = 2

def testband():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3',  ecg=True,  imu=True, ppg=True, eda=True, emg=True,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    #odh is your data
    odh.visualize(num_samples=10000)

def preparemodel():
    regex_filters = [
    data_handler.RegexFilter(left_bound = "C_", right_bound="_R_", values = [str(i) for i in REPS], description='reps'),
    data_handler.RegexFilter(left_bound = "_R_", right_bound="_emg.csv", values = [str(i) for i in CLASSES], description='classes')
]
    odh = data_handler.OfflineDataHandler()
    odh.get_data(folder_location="data/S" + str(0) + "/", regex_filters= regex_filters)
    
    windows, metadata = odh.parse_windows(WINDOW_SIZE,WINDOW_INC)
    fe = feature_extractor.FeatureExtractor()
    feature_dic = {}
    feature_dic['training_features'] = fe.extract_feature_group("HTD", windows)
    feature_dic['training_labels'] = metadata['classes']
    model = emg_predictor.EMGClassifier("LDA")
    model.fit(feature_dictionary=feature_dic)
    model.add_velocity(windows, metadata['classes'])
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', ecg=True, imu=True, ppg=True, eda=True, emg=True,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    feature_list = feature_extractor.FeatureExtractor().get_feature_groups()['HTD']
    oc = emg_predictor.OnlineEMGClassifier(model, WINDOW_SIZE, WINDOW_INC, odh, feature_list, std_out=True)
    oc.run(True)
    

def collectdata():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', ecg=True, imu=True, ppg=True, eda=True, emg=True,filtering=True,emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    #odh is your data
    # odh.visualize_channels(list(range(8)), num_samples=10000)
    
    args = {
        "media_folder": "images/",
        "data_folder": "data/S" + str(0) + "/",
        "num_reps": 6,
        "rep_time": 5,
        "rest_time": 1,
        "auto_advance": True
    }
    guii = gui.GUI(odh, args=args, debug=False)
    guii.download_gestures([1,2,3,4,5], "images/")
    guii.start_gui()

if __name__ == "__main__":
    if STAGE == 0:
        collectdata()
    if STAGE == 1:
        preparemodel()
    if STAGE == 2:
        testband()

    