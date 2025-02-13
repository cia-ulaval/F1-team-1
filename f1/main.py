import time
from drive import *

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
    drive()