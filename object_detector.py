from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from kalman_filter import KalmanFilter

# Load the YOLOv8 model
model = YOLO('models/yolov8n.pt')

# Open the video file
video_path = "videos/vehicle-counting.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
opencv_trackers = {}
first_detection_without_id = True

kf = KalmanFilter()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame, (1000, 800))

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker='bytetrack.yaml')

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes is not None and hasattr(results[0].boxes, 'id'):
            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            except AttributeError:
                track_ids = None
        else:
            print("No boxes found in the results")
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if track_ids is not None:
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]

                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    
        for i, box in enumerate(boxes):
            x, y, w, h = box
            x, y, w, h = float(x), float(y), float(w), float(h)
            center_point = ((x + w / 2), (y + h / 2))

            predicted = kf.predict(center_point[0], center_point[1])
            pred_x, pred_y = int(predicted[0][0]), int(predicted[1][0])
    
            # Draw a circle on the predicted coordinates
            cv2.circle(annotated_frame, (pred_x, pred_y), 20, (255, 0, 0), 4)

        # Display the annotated frame
        cv2.imshow("Task1", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()