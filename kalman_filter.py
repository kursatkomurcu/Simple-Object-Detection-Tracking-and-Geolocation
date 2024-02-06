import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        
        # Initialize state (x, y, delta x, delta y)
        self.kf.statePre = np.zeros((4, 1), dtype=np.float32)

    def predict(self, coordX=None, coordY=None):
        ''' Predict the next state (x, y) with the Kalman filter. '''
        if coordX is not None and coordY is not None:
            measurement = np.array([[np.float32(coordX)], [np.float32(coordY)]])
            self.kf.correct(measurement)

        # Predict the next state
        prediction = self.kf.predict()
        x, y = prediction[0], prediction[1]

        return x, y
