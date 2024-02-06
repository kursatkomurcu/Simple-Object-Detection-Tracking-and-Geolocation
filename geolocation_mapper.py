import cv2
import numpy as np

def extract_frames(video_path, interval=1):
    """ Extract frames from a video at a given interval. """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frames = []
    
    while success:
        if count % interval == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
        
    vidcap.release()
    return frames

def detect_and_match_features(image1, image2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Create matcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = matcher.match(descriptors1, descriptors2)

    # Ensure 'matches' is a list so it can be sorted
    matches = list(matches)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2, matches


def find_drone_position(points1, points2):
    # Find homography
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # The drone's position in its own frame is typically the center of the image
    drone_position_in_frame = (points1[0][0], points1[0][1])
    
    # Use homography to transform the drone's position in the frame to the satellite image's perspective
    drone_position_in_satellite = cv2.perspectiveTransform(np.array([[drone_position_in_frame]], dtype='float32'), homography)
    
    return drone_position_in_satellite[0][0]

def calculate_azimuth(positions):
    """
    Calculate the azimuth from sequential positions.
    positions should be a list of (x, y) tuples.
    """
    if len(positions) < 2:
        return None
    
    delta_y = positions[-1][1] - positions[-2][1]
    delta_x = positions[-1][0] - positions[-2][0]
    angle_radians = np.arctan2(delta_y, delta_x)
    angle_degrees = np.degrees(angle_radians)
    
    # Convert angle range from (-180, 180) to (0, 360)
    azimuth = (angle_degrees + 360) % 360
    
    return azimuth    

if __name__ == '__main__':
    video_path = 'videos/drone.mp4'
    satellite_image_path = 'drone-img.png'
    interval = 30

    # Read the satellite image
    satellite_image = cv2.imread(satellite_image_path)
    
    # Extract frames from the video
    frames = extract_frames(video_path, interval)
    
    # List to store drone positions
    drone_positions = []
    
    # Process each frame
    for i, frame in enumerate(frames):
        # Detect and match features
        points1, points2, matches = detect_and_match_features(frame, satellite_image)
        
        # Find drone position
        if len(matches) >= 4:  # Homography requires at least 4 matches
            position = find_drone_position(points1, points2)
            drone_positions.append(position)
            print(f"Frame {i}: Drone Position: {position}")
        else:
            print(f"Frame {i}: Not enough matches to find position")
    
    # Calculate azimuth if possible
    if len(drone_positions) > 1:
        azimuth = calculate_azimuth(drone_positions)
        print(f"Azimuth: {azimuth} degrees")
    else:
        print("Not enough positions to calculate azimuth")
