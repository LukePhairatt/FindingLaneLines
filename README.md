In submission to Finding Lane Line project- Self driving car course at Udacity

- Run LaneDetection_pipline.py for image inputs
- Run LaneDetection_videoFinal.py for video inputs
- Run LaneDetection_videoFinal_challenge.py for the extra challenge video

Parameters are tuned for these images/video
See

vertices = np.array([[...]]) in process_image()

and

lines_xy = FindLineROI_XY(mean_lines, [xxx, img.shape[0]], img.shape) in hough_line()
