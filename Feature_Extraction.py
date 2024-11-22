import cv2
from matplotlib import pyplot as plt

# Image To Feature Extraction Method

def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    
    
    # Initiate ORB detector
    ORB = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp, des = ORB.detectAndCompute(image, None)
    
    
    '''
    # Initiate SIFT detector
    SIFT = cv2.SIFT_create()
    
    # Fine the keypoints and descriptors with SIFT
    kp, des = SIFT.detectAndCompute(image, None)
    '''
    
    '''
    # Initiate BRISK detector
    BRISK = cv2.BRISK_create()

    # Fine the keypoints and descriptors with SURF
    kp, des = BRISK.detectAndCompute(image, None)
    '''

    return kp, des

def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.imshow(display)
    plt.show()


def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images
    
    """
    kp_list = []
    des_list = []
    
    for image in images:
        kp, des = extract_features_function(image)
        kp_list.append(kp)
        des_list.append(des)

    
    return kp_list, des_list