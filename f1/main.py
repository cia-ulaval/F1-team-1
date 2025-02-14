import time
from drive import *
from stream import *
import subprocess 
from pynput.keyboard import Controller , Key

def cli():
    print('hello world')
"""
# --- Lancement de TrackMania ---
def launch_trackmania():
    trackmania_shortcut = r"C:\Users\gueid\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Trackmania.url"
    
    try:
        subprocess.Popen(["cmd", "/c", "start", "", trackmania_shortcut], shell=True)
        print("TrackMania lancé avec succès.")
    except Exception as e:
        print("Erreur lors du lancement de TrackMania:", e)

# --- Simulation de la touche Enter via pynput ---
def press_enter(count=3):
    keyboard = Controller()
    for i in range(count):
        print(f"Appui sur Enter ({i+1}/{count})")
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        time.sleep(0.5)

def press_down(count=1):
    keyboard = Controller()
    for i in range(count):
        print(f"Appui sur Enter ({i+1}/{count})")
        keyboard.press(Key.down)
        keyboard.release(Key.down)
        time.sleep(0.5)
"""

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