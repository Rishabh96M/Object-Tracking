# Q2.py
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Answers for Q2 of first assignment

import matplotlib.pyplot as plt
import estimations
import numpy as np
import cv2


def traajectory_of_ball(vid_name):
    """
    Definition
    ---
    Method to extract top and bottom points of a ball in each frame.

    Parameters
    ---
    vid_name : path and name of video file

    Returns
    ---
    x : list of x poistions of top and bottom pixel in each frame
    y : list of y poistions of top and bottom pixel in each frame
    """
    # Opening video file
    cap = cv2.VideoCapture(vid_name)

    # initialising empty lists
    x = []
    y = []

    # Reading the video
    while (cap.isOpened()):
        ret, frame = cap.read()
        # Checking for frame
        if ret:
            red = frame[:, :, 2]  # Extracting Red Frame

            # Inverse Thresholding to convert to binary
            ret, thresh = cv2.threshold(red, 230, 255, cv2.THRESH_BINARY_INV)
            row, col = np.where(thresh == 255)  # Positions of ball in frame

            # Top and Bottom points of ball in frame
            top = [int(np.average(col)), np.min(row)]
            bottom = [int(np.average(col)), np.max(row)]

            x.append(top[0])
            y.append(top[1])
            x.append(bottom[0])
            y.append(bottom[1])

            # Plotting top and bottom points in each frame
            cv2.circle(frame, top, 8, (0, 255, 0), -1)
            cv2.circle(frame, bottom, 8, (0, 255, 0), -1)
            cv2.imshow('frame', frame)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            break
    # Closing all the handelers
    cap.release()
    cv2.destroyAllWindows()
    return x, y


if __name__ == '__main__':
    x, y = traajectory_of_ball('Videos/ball_video1.mp4')
    y_new = estimations.ols(x, y, 2)
    plt.figure()
    plt.title('Without Noise')
    plt.plot(x, y, 'ro', label='poistion of ball')
    plt.xlabel('time')
    plt.ylabel('ball position')
    plt.plot(x, y_new, 'g--', label='OLS')
    plt.gca().invert_yaxis()
    plt.legend()

    x, y = traajectory_of_ball('Videos/ball_video2.mp4')
    y_new = estimations.ols(x, y, 2)
    plt.figure()
    plt.title('With Noise')
    plt.plot(x, y, 'ro', label='poistion of ball')
    plt.xlabel('time')
    plt.ylabel('ball position')
    plt.plot(x, y_new, 'g--', label='OLS')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
