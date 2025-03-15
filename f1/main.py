
from libemg import streamers, data_handler, filtering, gui, emg_predictor, feature_extractor, utils
import keyboard
import vgamepad as vg 
import time

from drive import accelerate, brake, steer_left, steer_right, reset_controls
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
        
        if data[0]['imu'][0][0] >= 6.55:
            #pyautogui.press('right')
            #gamepad.right_trigger(255)
            #gamepad.update()

            accelerate()
        if data[0]['imu'][0][0] <= -6.55:
            #pyautogui.press('right')
            #gamepad.right_trigger(255)
            #gamepad.update()

            brake()

        if data[0]['imu'][0][2] >= 7.55:
            
            steer_right()
        if data[0]['imu'][0][2] <= -7.55:
            steer_left()
           
        else:
            time.sleep(0.1)
            reset_controls()
            


    
            
if __name__ == "__main__":
    cli()