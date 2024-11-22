import sys

import numpy as np
import cv2
from matplotlib import pyplot as plt


from m2bk import *
from Feature_Extraction import *
from Feature_Matching import *
from Trajectory_Estimation import *






if __name__=='__main__':
    np.random.seed(1)
    np.set_printoptions(threshold=sys.maxsize)


    # Data Load  
    dataset_handler = DatasetHandler()
    
    '''
    # frame Visualization

    image = dataset_handler.images[30]

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image, cmap='gray')
    plt.show()


    image_rgb = dataset_handler.images_rgb[30]

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image_rgb)
    plt.show()


    depth = dataset_handler.depth_maps[30]

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(depth, cmap='jet')
    plt.show()

    dataset_handler.k

    # Number of frames in the dataset

    print(dataset_handler.num_frames)

    image = dataset_handler.images[30]

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image, cmap='gray')
    plt.show()
    '''

    # Single Image To Feature Extraction  

    i = 1
    image = dataset_handler.images[i]
    kp, des = extract_features(image)
    print("Number of features detected in frame {0}: {1}\n".format(i, len(kp)))
    print("Coordinates of the first keypoint in frame {0}: {1}".format(i, str(kp[0].pt)))

    # Each Image To Features Extraction

    images = dataset_handler.images
    kp_list, des_list = extract_features_dataset(images, extract_features)

    i = 5
    print("Number of features detected in frame {0}: {1}".format(i, len(kp_list[i])))
    print("Coordinates of the first keypoint in frame {0}: {1}\n".format(i, str(kp_list[i][0].pt)))

    # Remember that the length of the returned by dataset_handler lists should be the same as the length of the image array
    print("Length of images array: {0}".format(len(images)))





    # Feature Matching
    i = 0 
    des1 = des_list[i]
    des2 = des_list[i+1]

    match = match_features(des1, des2)
    print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(match)))

    # Remember that a matcher finds the best matches for EACH descriptor from a query set


    # Visualize n first matches, set n to None to view all matches
    
    # set filtering to True if using match filtering, otherwise set to False
    n = 20
    filtering = False

    i = 0 
    image1 = dataset_handler.images[i]
    image2 = dataset_handler.images[i+1]

    kp1 = kp_list[i]
    kp2 = kp_list[i+1]

    des1 = des_list[i]
    des2 = des_list[i+1]

    match = match_features(des1, des2)
    if filtering:
        dist_threshold = 10
        match = filter_matches_distance(match, dist_threshold)

    image_matches = visualize_matches(image1, kp1, image2, kp2, match[:n])


    matches = match_features_dataset(des_list, match_features)

    i = 5
    print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(matches[i])))

    # Optional
    dist_threshold = 10

    filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)

    if len(filtered_matches) > 0:

        # Make sure that this variable is set to True if you want to use filtered matches further in your assignment
        is_main_filtered_m = False
        if is_main_filtered_m: 
            matches = filtered_matches

        i = 0
        print("Number of filtered matches in frames {0} and {1}: {2}".format(i, i+1, len(filtered_matches[i])))



    # Trajectory Estimation 

    i = 0
    match = matches[i]
    kp1 = kp_list[i]
    kp2 = kp_list[i+1]
    k = dataset_handler.k
    depth = dataset_handler.depth_maps[i]

    rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth1=depth)

    # print("Estimated rotation:\n {0}".format(rmat))
    # print("Estimated translation:\n {0}".format(tvec))

    i=30
    image1  = dataset_handler.images_rgb[i]
    image2 = dataset_handler.images_rgb[i + 1]

    image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
    plt.figure(figsize=(16, 12), dpi=100)
    plt.imshow(image_move)
    plt.show()


    image_move = visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=True)
    plt.figure(figsize=(16, 12), dpi=100)
    plt.imshow(image_move)
    plt.show()
    # These visualizations might be helpful for understanding the quality of image points selected for the camera motion estimation


    '''
    depth_maps = dataset_handler.depth_maps
    trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)

    i = 1
    print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))

    # Remember that the length of the returned by trajectory should be the same as the length of the image array
    print("Length of trajectory: {0}".format(trajectory.shape[1]))
    '''

    # Note: Make sure to uncomment the below line if you modified the original data in any ways
    #dataset_handler = DatasetHandler()


    # Part 1. Features Extraction
    images = dataset_handler.images
    kp_list, des_list = extract_features_dataset(images, extract_features)


    # Part II. Feature Matching
    matches = match_features_dataset(des_list, match_features)

    # Set to True if you want to use filtered matches or False otherwise
    is_main_filtered_m = True
    if is_main_filtered_m:
        dist_threshold = 100
        filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)
        matches = filtered_matches


    # Part III. Trajectory Estimation
    depth_maps = dataset_handler.depth_maps
    trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, )


    #!!! Make sure you don't modify the output in any way
    # Print Submission Info
    print("Trajectory X:\n {0}".format(trajectory[0,:].reshape((1,-1))))
    print("Trajectory Y:\n {0}".format(trajectory[1,:].reshape((1,-1))))
    print("Trajectory Z:\n {0}".format(trajectory[2,:].reshape((1,-1))))    


    visualize_trajectory(trajectory)


    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    
    locX, locY, locZ = [], [], []
    
    # Set up the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 3)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')
    
    # Determine plot limits
    max_value = np.max(trajectory)
    min_value = np.min(trajectory)
    maxY, minY = max_value, min_value
    
    def animate(i):
        current_pos = trajectory[:, i]
        print(f"Frame {i}: {current_pos}")  # 프레임 번호와 현재 포지션 출력
        locX.append(current_pos[0])
        locY.append(current_pos[1])
        locZ.append(current_pos[2])
    
        traj_main_plt.clear()
        ZY_plt.clear()
        YX_plt.clear()
        D3_plt.clear()
    
        # Main trajectory plot (Z, X)
        traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=1.06)
        traj_main_plt.plot(locZ, locX, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4)
        traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
        traj_main_plt.set_xlim([min_value, max_value])
        traj_main_plt.set_ylim([min_value, max_value])
        traj_main_plt.set_xlabel("Z")
        traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)
    
        # ZY plot
        ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
        ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
        ZY_plt.set_xlim([min_value, max_value])
        ZY_plt.set_ylim([minY, maxY])
        ZY_plt.set_ylabel("Y")
        ZY_plt.axes.xaxis.set_ticklabels([])
    
        # YX plot
        YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
        YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
        YX_plt.set_xlim([minY, maxY])
        YX_plt.set_ylim([min_value, max_value])
        YX_plt.set_xlabel("Y")
        YX_plt.set_ylabel("X")
    
        # 3D plot
        D3_plt.plot3D(locX, locZ, locY, zorder=0)
        D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
        D3_plt.set_xlim3d(min_value, max_value)
        D3_plt.set_ylim3d(min_value, max_value)
        D3_plt.set_zlim3d(min_value, max_value)
        D3_plt.set_xlabel("X", labelpad=0)
        D3_plt.set_ylabel("Z", labelpad=0)
        D3_plt.set_zlabel("Y", labelpad=-2)
        D3_plt.view_init(45, azim=30)
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=trajectory.shape[1], interval=100, repeat=True)
    HTML(ani.to_jshtml())
    plt.tight_layout()
    plt.show()
