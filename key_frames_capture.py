import cv2
import numpy as np

def setup_video_sift_flann(filename):
    # Initialize video capture and SIFT detector
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")
    sift = cv2.SIFT_create()

    # Setup FLANN matcher with KDTree algorithm
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    return cap, sift, flann

def find_good_matches(sift, flann, img1, img2):
    # Detect features and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match descriptors using FLANN and apply Lowe's ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < 0.75 * n.distance], kp1, kp2

def key_frames_capture(filename):
    cap, sift, flann = setup_video_sift_flann(filename)
    ret, base_img = cap.read()
    if not ret:
        print("Failed to read the first frame from the video")
        cap.release()
        return

    print("Captured frame0.jpg")
    cv2.imwrite('./keyframes/frame0.jpg', base_img)

    frame_index, count = 1, 1
    key_frame_interval = 30  # Process every 30 frames
    min_match_num = 100  # minimum number of matches required
    max_match_num = 8000  # maximum number of matches

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % key_frame_interval == 0:
            good_matches, kp1, kp2 = find_good_matches(sift, flann, base_img, frame)
            if len(good_matches) > 4:
                # Check if there are enough good matches to construct a homography matrix
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if min_match_num < np.count_nonzero(mask) < max_match_num:
                    print(f"Captured frame{count}.jpg")
                    cv2.imwrite(f'./keyframes/frame{count}.jpg', frame)
                    count += 1
            base_img = frame  # Update reference frame regardless of whether it was saved


        frame_index += 1

    cap.release()

if __name__ == "__main__":
    key_frames_capture('test2.mp4')
