
from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils
import keyboard
import vgamepad as vg 
import time
gamepad = vg.VX360Gamepad()


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
        data = odh.get_data(2)

        
        print(data[0]['imu'][0])
        print("-----------------------------")
        
        if data[0]['imu'][0][0] < 5000000000:
            #pyautogui.press('right')
            gamepad.right_trigger(255)
            gamepad.update()
            time.sleep(0.1)
        if data[0]['imu'][0][1] < 500000000000:
            #pyautogui.press('left')
            gamepad.left_joystick(x_value=-32000 , y_value= 0)
            gamepad.update()
            time.sleep(0.1)
        
            
if __name__ == "__main__":
    cli()