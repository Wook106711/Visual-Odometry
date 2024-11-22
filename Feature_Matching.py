import cv2
from matplotlib import pyplot as plt

def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    
    
    # Create ORB or BRISK`s BFMatcher object (ORB => Hamming Distance)
    BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = BF.match(des1, des2)

    # Sort them in the order of their distance.
    match = sorted(matches, key=lambda x: x.distance)
    
    
    '''
    # FLANN Matching Parameter 
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)  

    # Create SIFT or BRISK`s FLANNMatcher object 
    FLANN = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors.
    knn_matches = FLANN.knnMatch(des1, des2, k=2)

    # Ratio Test
    match = []
    for m, n in knn_matches:
        if m.distance < 0.7 * n.distance:  
            match.append(m)
    '''

    return match


# Optional
def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []

    min_distance = max(match[0].distance, 1e-6)  # Best match distance
    for m in match:
        if m.distance <= dist_threshold * min_distance:
            filtered_match.append(m)
    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    plt.show()


def match_features_dataset(des_list, match_features):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
               
    """
    matches = []
    
    for i in range(len(des_list) - 1):
        match = match_features(des_list[i], des_list[i + 1])
        matches.append(match)
    
    return matches


# Optional
def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold
               
    """
    filtered_matches = []
    
    for match in matches:
        filtered = filter_matches_distance(match, dist_threshold)
        filtered_matches.append(filtered)
    
    return filtered_matches