import cv2
import numpy as np

def stitch_blend(img1, img2, H):

    # Get dimensions of the input images
    h1, w1 = img1.shape[:2]
    h2, w2, depth = img2.shape

    # Define corners of the original images
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transform corners of img1 using the homography matrix
    transformed_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    # Determine the size of the output canvas based on transformed coordinates
    combined_corners = np.concatenate((corners_img2, transformed_corners_img1), axis=0)
    [x_min, y_min] = np.int32(combined_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(combined_corners.max(axis=0).ravel() + 0.5)
    output_size = (x_max - x_min, y_max - y_min)

    # The output matrix after affine transformation
    offset_transform = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp img1 using the composite transformation matrix
    img_output = cv2.warpPerspective(img1, offset_transform.dot(H), output_size)


    '''
    Average Blending: Average blending combines the pixel values from overlapping areas 
        of two images by calculating the simple average,
        resulting in a direct blend without emphasizing either image.
    
    '''
    # for i in range(-y_min, h2 - y_min):
    #     for j in range(-x_min, w2 - x_min):
    #         # Check if the pixel in img2 at this position contains color information (not black)
    #         if np.any(img2[i + y_min][j + x_min]):
    #             if np.any(img_output[i][j]):
    #                 # If the corresponding pixel in the output image also has color information,
    #                 # average the color values of img1 and img2
    #                 for k in range(depth):
    #                     img_output[i][j][k] = np.uint8(
    #                         (int(img2[i + y_min][j + x_min][k])
    #                          + int(img_output[i][j][k])) / 2)
    #             else:
    #                 # If the output image pixel is black, directly copy the pixel from img2
    #                 img_output[i][j] = img2[i + y_min][j + x_min]


    '''
    Linear Blending: Linear blending uses a weighted average of pixel values from two images, 
        applying a specified ratio to create a smooth transition between overlapping regions.
    '''
    # alpha = 0.8     # alpha controls the blending ratio, favoring img2 over img1
    # for i in range(-y_min, h2 - y_min):
    #     for j in range(-x_min, w2 - x_min):
    #         # Check for color information in the pixel from img2
    #         if np.any(img2[i + y_min][j + x_min]):
    #             # Blend the pixel values if the corresponding pixel in the output image has color
    #             if np.any(img_output[i][j]):
    #                 img_output[i][j] = (alpha * img2[i + y_min][j + x_min] + (1 - alpha) * img_output[i][j])
    #             else:
    #                 # Copy the pixel from img2 if the corresponding pixel in the output is black
    #                 img_output[i][j] = img2[i + y_min][j + x_min]

    return img_output



def stitch(img1, img2, H):
 
    # Get dimensions of the input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define corners of the original images
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transform corners of img1 using the homography matrix
    transformed_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    # Determine the size of the output canvas based on transformed coordinates
    combined_corners = np.concatenate((corners_img2, transformed_corners_img1), axis=0)
    [x_min, y_min] = np.int32(combined_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(combined_corners.max(axis=0).ravel() + 0.5)
    output_size = (x_max - x_min, y_max - y_min)

    # The output matrix after affine transformation
    offset_transform = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp img1 using the composite transformation matrix
    transformed_img1 = cv2.warpPerspective(img1, offset_transform.dot(H), output_size)

    # Create output image and place img2 within the new canvas
    output_image = transformed_img1.copy()
    for y in range(h2):
        for x in range(w2):
            if all(img2[y, x] != 0):  # Check if the pixel is not completely black
                output_image[y - y_min, x - x_min] = img2[y, x]

    return output_image


# Mathematicals are from https://www.cnblogs.com/cheermyang/p/5431170.html
def cylindrical_project(img, f=550):

    height, width, depth = img.shape

    # Initialize the output image with zeros (all black)
    cylindrical_img = np.zeros_like(img)
    
    # Calculate the center coordinates of the image
    centerX = width / 2
    centerY = height / 2

    # Iterate over each pixel in the cylindrical image
    for i in range(width):

        # Calculate the angle theta from the center for the current column
        theta = (i - centerX) / f

        # Determine the corresponding x-coordinate in the input image
        pointX = int(f * np.tan(theta) + centerX)

        # Calculate the scaling factor for y-coordinate adjustment due to the cylindrical projection
        pointY_factor = 1 / np.cos(theta)


        # Iterate over each row in the output image
        for j in range(height):

            # Determine the corresponding y-coordinate in the input image
            pointY = int((j - centerY) * pointY_factor + centerY)

            # Check bounds and assign values
            if 0 <= pointX < width and 0 <= pointY < height:
                # If within bounds, copy the pixel values from the input image
                for k in range(depth):
                    cylindrical_img[j, i, k] = img[pointY, pointX, k]
            else:
                # Set to black if out-of-bounds; though np.zeros_like already set it to 0
                cylindrical_img[j, i, :] = 0

    return cylindrical_img


# Inspired from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def compute_homography(img1, img2):

    # Call the SIFT method
    sift = cv2.SIFT_create()

    # Get dimensions
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    # Crop the right part of img1 for detecting SIFT
    img1_crop = img1[:, w1 - w2:]  

    # Adjust coordinates to original scale if img1 was cropped
    diff = w1 - img1_crop.shape[1]

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_crop, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match descriptors using Brute-Force matcher
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)  # Using L2 (Euclidean) distance
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe's ratio test
    match_ratio = 0.6
    valid_matches = [m1 for m1, m2 in matches if m1.distance < match_ratio * m2.distance]

    # Require at least 4 matches to compute homography
    if len(valid_matches) >= 4:
        # Extract the coordinates of matching points
        img1_pts = np.float32([kp1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
        img2_pts = np.float32([kp2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
        img1_pts[:, :, 0] += diff  # Adjust coordinates back to original if cropped

        # Compute the Homography matrix
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # Redo homography computation to remove outliers
            img1_pts = img1_pts[mask.ravel() == 1]
            img2_pts = img2_pts[mask.ravel() == 1]
            H, _ = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

            # Draw matches
            # draw_img = cv2.drawMatches(img1, kp1, img2, kp2, 
            #         [valid_matches[i] for i in range(len(valid_matches)) if mask[i]], 
            #         None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imwrite('matching points.jpg', draw_img)

            return H
    else:
        print("Not enough matches to compute homography.")
        return None


def crop_result(result_img):
    
    # Determine the height and width of the input image
    h = result_img.shape[0]
    w = result_img.shape[1]

    # Set the percentage of the original width to retain
    cut_off_percent = 0.95

    # Calculate height bounds for cropping
    h1 = int(h * 0.05)      # Start height (5% from top)
    h2 = int(h * 0.95)      # End height (95% from top, i.e., crops 5% from bottom)
    w_new = int(w * cut_off_percent) # New width after cropping 10% from the right

    # Crop the image according to calculated dimensions
    result_img = result_img[h1:h2, :w_new]

    return result_img

def cut_corners(img):

    # Extract dimensions of the original image
    h, w = img.shape[:2]

    # Percentage of each edge to be removed
    percent = 0.07

    # Calculate new dimensions for the cropped image
    new_width = int(w * (1 - 2 * percent))  # Reduce width by 2*percent
    newheight = int(h * (1 - 2 * percent))  # Reduce height by 2*percent

    # Calculate the starting points for the new cropped area
    height_start = int(h * percent)         # Start cropping 8% from the top
    width_start = int(w * percent)          # Start cropping 8% from the left

     # Crop the image to the new dimensions
    new_img = img[height_start: height_start + newheight, width_start: width_start + new_width]
    return new_img


def stitch_image(img1_path, img2_path):
    # Input images
    if isinstance(img1_path, str):
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (640, 360))
        img1 = cylindrical_project(img1)
    else:
        img1 = img1_path

    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (640, 360))
    img2 = cylindrical_project(img2)

    H = compute_homography(img1, img2)

    stitched_image = stitch(img1, img2, H)
    # stitched_image = stitch_blend(img1, img2, H)

    stitched_image = crop_result(stitched_image)

    return stitched_image


if __name__ == "__main__":
    
    current_img = 'keyframes/frame0.jpg'

    key_frame_num = 5
    for i in range(1, key_frame_num):
        next_img = 'keyframes/frame{}.jpg'.format(i)
        print("Stitching frame{} and frame{}...".format(i - 1, i))
        current_img = stitch_image(current_img, next_img)

    result = current_img
    result = cut_corners(result)
    cv2.imwrite('panorama.jpg', result)
    print("panoramic.jpg complete!")