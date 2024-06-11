# Import the necessary libraries
import numpy as np
import argparse
import cv2 
import time

# Time before script execution
start_time = time.time()

# Construct the argument parse 
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: MobileNetSSD_deploy.prototxt for Caffe model')
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: MobileNetSSD_deploy.caffemodel for Caffe model')
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

# Check if the video was opened correctly
if not cap.isOpened():
    raise ValueError("Video open failed. Please check your camera or video file path.")

# Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

# Determine frame rate to process
input_fps = cap.get(cv2.CAP_PROP_FPS)
desired_fps = 10
frame_skip = round(input_fps / desired_fps)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, desired_fps, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
total_proc_time = 0
fps_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 'frame_skip' frame
    if frame_count % frame_skip == 0:
        fps_count+=1
        start_time = time.time()
        frame_resized = cv2.resize(frame, (300, 300)) # resize frame for prediction

        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        net.setInput(blob)
        detections = net.forward()

        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                # Scale coordinates back to the original frame size
                heightFactor = frame.shape[0] / 300.0
                widthFactor = frame.shape[1] / 300.0
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)

                # # Draw the bounding box and label
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(round(confidence, 2))
                    print(label)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    

           
         # Calculate processing time and accumulate
        proc_time = time.time() - start_time
        total_proc_time += proc_time

        # Calculate the effective frame rate
        if total_proc_time > 0:
            effective_fps = (fps_count + 1) / total_proc_time
            print(f"Effective processing frame rate: {effective_fps:.2f} FPS")


        # Write the processed frame to the output
        out.write(frame)
    
    frame_count += 1

# Release everything when job is finished
cap.release()
out.release()
# cv2.destroyAllWindows()

# Time after script execution
end_time = time.time()

# Lapsed time calculation
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
