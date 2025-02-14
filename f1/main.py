import time
from drive import *
from stream import *
import subprocess 
from pynput.keyboard import Controller , Key

def cli():
    print('hello world')

def drive():
    print("🚗 Départ en course...")
    time.sleep(10)
    
    accelerate()
    print("Accélération...")
    time.sleep(10)

    steer_left()
    print("Tourne à gauche...")
    time.sleep(5)

    steer_right()
    print("Tourne à droite...")
    time.sleep(5)

    brake()
    print("Freinage...")
    time.sleep(10)

    reset_controls()
    print("Toutes les commandes sont relâchées.")

if __name__ == "__main__":
    cli()
    """
    launch_trackmania()
    time.sleep(10)
    press_enter(4)
    time.sleep(10)
    press_enter(3)
    press_down(1)
    press_enter(1)
    """
    drive()   