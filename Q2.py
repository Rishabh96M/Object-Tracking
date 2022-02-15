# Object-Tracking
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Finding the top and bottom of the red ball in each frame and
#              performing least squares for estimating the path of the ball

import matplotlib.pyplot as plt
import estimations
import numpy as np
import cv2


def traajectory_of_ball(vid_name):
    cap = cv2.VideoCapture(vid_name)
    x = []
    y = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            red = frame[:, :, 2]
            ret, thresh = cv2.threshold(red, 230, 255, cv2.THRESH_BINARY_INV)
            x_1, y_1, w, h = cv2.boundingRect(thresh)
            row, col = np.where(thresh == 255)

            top = [int(np.average(col)), np.min(row)]
            bottom = [int(np.average(col)), np.max(row)]

            x.append(top[0])
            y.append(top[1])
            x.append(bottom[0])
            y.append(bottom[1])

            cv2.circle(frame, top, 8, (0, 255, 0), -1)
            cv2.circle(frame, bottom, 8, (0, 255, 0), -1)

            cv2.imshow('frame', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return x, y


if __name__ == '__main__':
    x, y = traajectory_of_ball('Videos/ball_video1.mp4')
    plt.figure()
    plt.title('Without Noise')
    plt.plot(x, y, 'ro', label='poistion of ball')
    y_new = estimations.ols(x, y, 2)
    plt.plot(x, y_new, 'g-', label='OLS')
    plt.gca().invert_yaxis()
    plt.legend()

    x, y = traajectory_of_ball('Videos/ball_video2.mp4')
    plt.figure()
    plt.title('With Noise')
    plt.plot(x, y, 'ro', label='poistion of ball')
    y_new = estimations.ols(x, y, 2)
    plt.plot(x, y_new, 'g-', label='OLS')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
