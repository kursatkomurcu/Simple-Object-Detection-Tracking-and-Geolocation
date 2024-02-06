## Object Detection and Tracking

Yolov8n pretrained model was used for object detection and tracking. After detecting an object, the model gives it an id and tracks it. Also the kalman filter was added for tracking and prediction of locations of objects. When Yolo model can detect an object but cannot track it, the kalman filter can still track the object.

```bash
python object_detector.py
```

![track](https://github.com/kursatkomurcu/Simple-Object-Detection-Tracking-and-Geolocation/blob/main/track.png)

## Geolocation Mapper

These code finds drone coordinates using drone video and satellite image. An video example and an image from this video was used for experiment. **ORB Detector** was used for finding key points of images. After that these key points was matched.  Then points was gotten.

Secondly, drone coordinates were found by using **findHomography** and **perspectiveTransform** functions and the azimuth calculated like below:

```python
delta_y = positions[-1][1] - positions[-2][1]
delta_x = positions[-1][0] - positions[-2][0]
angle_radians = np.arctan2(delta_y, delta_x)
angle_degrees = np.degrees(angle_radians)
    
# Convert angle range from (-180, 180) to (0, 360)
azimuth = (angle_degrees + 360) % 360
```

The output of drone coordinates are [y, x]

```bash
python geolocation_mapper.py
```

![track](https://github.com/kursatkomurcu/Simple-Object-Detection-Tracking-and-Geolocation/blob/main/drone_position.png)

