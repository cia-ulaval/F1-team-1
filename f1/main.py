import sys
import argparse

from libemg.emg_predictor import EMGClassifier, OnlineEMGClassifier
from libemg.data_handler import OnlineDataHandler
from libemg import streamers

from gesture_mapper import map_class_to_action

from vgamepad_controller import handle_pyautogui_action
#from ros2_controller import create_ros_node, shutdown_ros_node

def main():
    parser = argparse.ArgumentParser(
        description="Run real-time classification and map predictions to either vgamepad or ros."
    )
    parser.add_argument("--mode", choices=["vgamepad","ros"], default="vgamepad",
                        help="Which control mode to use: vgamepad or ros")
    parser.add_argument("--model", default="my_pretrained_classifier.pkl",
                        help="Path to the pre-trained classifier file")
    args = parser.parse_args()

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
    print("Streaming from BioPoint device started...")

    odh = OnlineDataHandler(shared_mem)

    try:
        offline_classifier = EMGClassifier.from_file(args.model)
        print(f"Loaded classifier from {args.model}")
    except FileNotFoundError:
        print(f"Error: model file '{args.model}' not found. Exiting.")
        streamer.cleanup()
        sys.exit(1)

    WINDOW_SIZE = 50
    WINDOW_INCREMENT = 10
    feature_list = []

    classifier = OnlineEMGClassifier(
        offline_classifier,
        window_size=WINDOW_SIZE,
        window_increment=WINDOW_INCREMENT,
        online_data_handler=odh,
        feature_list=feature_list
    )

    ros_node = None
    if args.mode == "ros":
        import rclpy
        from ros_controller import GestureROSController
        rclpy.init()
        ros_node = GestureROSController()

    def on_prediction(pred_label, probabilities):
        action = map_class_to_action(pred_label)
        print(f"Predicted label={pred_label} => action={action}")

        if args.mode == "vgamepad":
            handle_pyautogui_action(action)
        elif args.mode == "ros":
            #ros_node.publish_action(action)
            pass

    print(f"Running in {args.mode.upper()} mode. Press Ctrl+C to stop.\n")

    try:
        classifier.run(block=True, callback=on_prediction)

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down classification & streamer...")
        classifier.stop_running()
        streamer.cleanup()
        if ros_node:
            ros_node.destroy_node()
            import rclpy
            rclpy.shutdown()
        print("Stopped. Goodbye!")

if __name__ == "__main__":
    main()
