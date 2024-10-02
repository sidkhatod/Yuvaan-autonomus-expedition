# Yuvaan-autonomus-expedition
In this project, I developed software for a Mars rover to autonomously navigate an arena with directional arrows.
A pretrained TensorFlow model, ResNet50, was used to recognize the directions from the arrows, while YOLO was employed to create bounding boxes around the arrows, providing the necessary coordinates. 
Using these coordinates, we estimated the distance to the arrows with a RealSense or ZED camera for depth sensing.
