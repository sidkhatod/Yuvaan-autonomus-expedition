import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import time
import pyzed.sl as sl

model1 = YOLO("best.pt")
filepath = "modelrestnet.h5"
model2 = tf.keras.models.load_model(filepath)


# depth and yolo
def depth():
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA for higher depth accuracy
    zed.open(init_params)

    depth_ar = []
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth_map = sl.Mat()

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)  # Get the left camera image
                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Get the depth map

                img = image.get_data()[:, :, :3]  # Remove the alpha channel
                img = np.asanyarray(img)

                results = model1(img)
                for result in results:
                    cv2.imshow('ZED Camera', result.plot())
                    cv2.waitKey(100)
                    cord = result.boxes.xyxy
                    if len(cord) != 0:
                        x1, y1 = cord[0][0], cord[0][1]
                        x2, y2 = cord[0][2], cord[0][3]
                        (x, y) = (x2 + x1) / 2, (y2 + y1) / 2

                        # Retrieve depth at the center of the detected object
                        z_depth = depth_map.get_value(int(x), int(y))[1]  # [1] gives the depth value
                        print(z_depth)
                        depth_ar.append(z_depth)

                        if z_depth <= 2.1:  # Check if object is within 2 meters
                            return z_depth
                    else:
                        continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Close the ZED camera
        zed.close()
        cv2.destroyAllWindows()
    return z_depth


# arrow direction detection
def detect():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA for higher depth accuracy
    zed.open(init_params)

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    def array2dir(array):
        none_prob = 0.4
        print("Model output array:", array[0][0], array[0][1], array[0][2], array[0][3])
        down_prob, left_prob, right_prob, up_prob = array[0][:4]
        if left_prob > right_prob and left_prob > up_prob and left_prob > down_prob and left_prob > none_prob:
            print("left")
        elif right_prob > left_prob and right_prob > up_prob and right_prob > down_prob and right_prob > none_prob:
            print("right")
        # elif up_prob > left_prob and up_prob > right_prob and up_prob > down_prob and up_prob > none_prob:
        #     print("up")
        # elif down_prob > left_prob and down_prob > right_prob and down_prob > up_prob and down_prob > none_prob:
        #     print("down")
        else:
            print("none")

    # Detection
    start_time = time.time()
    duration = 5  # Run for 15 seconds

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                img = image.get_data()[:, :, :3]
                cv2.imshow('ZED Camera', img)

                img = cv2.resize(img, (224, 224))
                img = np.expand_dims(img, axis=0)

                output = model2.predict(img)  # Perform prediction using the loaded model
                array2dir(output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Stop after the specified duration
            elapsed_time = time.time() - start_time
            if elapsed_time > duration:
                print(f"Stopping after {duration} seconds.")
                break

    finally:
        zed.close()
        cv2.destroyAllWindows()

    return output

def main():
    depth1 = depth()
    if depth1 <= 2.1:
        direction = detect()

    if direction == "left":
        print(-1)
        return -1
    elif direction == "right":
        print(1)
        return 1
    else:
        print(0)
        return 0

main()