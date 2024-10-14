import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import time
import pyrealsense2 as rs

model1=YOLO("bestv8.pt")
filepath = "modelrestnet.h5"
model2 = tf.keras.models.load_model(filepath)


#depth and yolo
def depth():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # Start streaming
    pipeline.start(config)
    depth_ar=[]
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            results = model1(img)
            for result in results:
                cv2.imshow('RealSense', result.plot())
                cv2.waitKey(100)
                cord = result.boxes.xyxy
                if(len(cord)!=0):
                    x1=cord[0][0]
                    x2=cord[0][2]
                    y1=cord[0][1]
                    y2=cord[0][3]
                    (x,y) = (x2 + x1)/2, (y2+y1)/2
                    zDepth = depth.get_distance(int(x),int(y))
                    print(zDepth)
                    depth_ar.append(zDepth)
                    if(zDepth<= 2):
                        return zDepth
                    #segmentation
                    # mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    # cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=-1)
                    # segmented_image = cv2.bitwise_and(img, img, mask=mask)
                    # cv2.imshow('Segmented Image', segmented_image)
                else:
                    continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
    return zDepth

#arrow direction detection
def detect():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    def array2dir(array):
        none_prob = 0.4
        print("Model output array:", array[0][0],array[0][1],array[0][2],array[0][3])  # Debugging line to see the output array
        down_prob, left_prob, rightprob, up_prob = array[0][:4]  # Adjust to take the first three values
        right_prob = rightprob
        if left_prob > right_prob and left_prob > up_prob and left_prob>down_prob and left_prob>none_prob:
            print("left")
        elif right_prob > left_prob and right_prob > up_prob and right_prob>down_prob and right_prob>none_prob:
            print("right")
        elif up_prob > left_prob and up_prob > right_prob and up_prob>down_prob and up_prob>none_prob:
            print("up")
        elif down_prob > left_prob and down_prob > right_prob and down_prob>up_prob and down_prob>none_prob:
            print("down")
        else:
            print("none")
    
    #detection 
    start_time = time.time()
    duration = 5  # Run for 5 seconds

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense', img)
            img = cv2.resize(img, (224, 224))  # Resize image
            img = np.asarray(img)  # Convert image to numpy array
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            output = model2.predict(img)  # Perform prediction using the loaded model
            array2dir(output)  # Call function to interpret the prediction

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Stop after the specified duration
            elapsed_time = time.time() - start_time
            if elapsed_time > duration:
                print(f"Stopping after {duration} seconds.")
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
    
    return output

depth1 = depth()
if(depth1<=2):
    direction = detect()
if(direction == "left"):
    out=-1
elif(direction == "right"):
    out=1
else:
    out=0
print(out)