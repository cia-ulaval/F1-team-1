from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils
import pyautogui

def cli():
    streamer, smm = streamers.sifi_biopoint_streamer(name='BioPoint_v1_3', 
                                                    ecg=True, 
                                                    imu=True, 
                                                    ppg=True, 
                                                    eda=True, 
                                                    emg=True,
                                                    filtering=True,
                                                    emg_notch_freq=60)
    odh = data_handler.OnlineDataHandler(smm)
    
    while True:
        data = odh.get_data(1)
        if data['Inertie'][0] < 50:
            pyautogui.press('right')
        elif data['Inertie'][0] < 50:
            pyautogui.press('left')

if __name__ == "__main__":
    cli()