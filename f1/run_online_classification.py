import time
import sys
import signal

from libemg import streamers
from libemg.data_handler import OnlineDataHandler
from libemg.emg_predictor import EMGClassifier, OnlineEMGClassifier

def main():
    """
    1) Start streaming from BioPoint device
    2) Create OnlineDataHandler
    3) Load a pre-trained classifier (my_pretrained_classifier.pkl)
    4) Create OnlineEMGClassifier => run
    """

    streamer, shared_mem = streamers.sifi_biopoint_streamer(
        name='BioPoint_v1_3',
        ecg=False,
        imu=False,
        ppg=False,
        eda=False,
        emg=True,
        filtering=True,
        emg_notch_freq=60
    )
    print("BioPoint streaming started...")

    odh = OnlineDataHandler(shared_mem)
    print("OnlineDataHandler created.")

    model_file = "my_pretrained_classifier.pkl"
    try:
        offline_classifier = EMGClassifier.from_file(model_file)
        print(f"Loaded pre-trained classifier from {model_file}.")
    except FileNotFoundError:
        print(f"Error: model file '{model_file}' not found. Exiting.")
        streamer.cleanup()
        sys.exit(1)

    WINDOW_SIZE = 50
    WINDOW_INCREMENT = 10

    feature_list = []

    classifier = OnlineEMGClassifier(
        offline_classifier,
        WINDOW_SIZE,
        WINDOW_INCREMENT,
        odh,
        feature_list
    )

    print("\nStarting real-time classification in blocking mode.")
    print("Press Ctrl+C to stop.\n")

    try:
        classifier.run(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping classification...")
        classifier.stop_running()
        streamer.cleanup()
        print("Classification and streaming stopped. Bye!")

if __name__ == "__main__":
    main()
