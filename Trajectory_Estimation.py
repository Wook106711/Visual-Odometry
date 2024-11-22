import numpy as np
import cv2
from matplotlib import pyplot as plt

def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
    for m in match:
        image1_points.append(kp1[m.queryIdx].pt)
        image2_points.append(kp2[m.trainIdx].pt)

    image1_points = np.array(image1_points, dtype=np.float32)
    image2_points = np.array(image2_points, dtype=np.float32)
    
    # Estimate the Essential matrix
    essential_matrix, mask = cv2.findEssentialMat(
        image1_points, 
        image2_points, 
        k, 
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=1.0
    )


    # Decompose the Essential matrix to obtain rotation and translation
    _, rmat, tvec, _ = cv2.recoverPose(
        essential_matrix, 
        image1_points, 
        image2_points, 
        k
    )
    
    
    return rmat, tvec, image1_points, image2_points


def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 
    
    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location
                  
                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven 
                  at the initialization of this function

    """
    trajectory = [np.zeros((3, 1))]  # Start with the origin point as a list element
    current_rotation = np.eye(3)     # Initial Rotation Matrix
    current_translation = np.zeros((3, 1))  # Initial position at the origin
    
    
    for i in range(len(matches)):
        
        kp1 = kp_list[i]
        kp2 = kp_list[i + 1]
        match = matches[i]

        # estimate the rmat, tvec
        rmat, tvec, _, _ = estimate_motion(match, kp1, kp2, k, depth_maps[i] if depth_maps else None)        
        
        # Update current rotation and translation
        
        # current_rotation = current_rotation @ rmat      # Update the cumulative rotation
        current_translation += rmat @ tvec  # Apply the current rotation to the translation
        
        # Append the new position to the trajectory list
        trajectory.append(current_translation.copy())
    
    # Convert trajectory to a numpy array of shape (3, len)
    trajectory = np.hstack(trajectory)
    
    return trajectory