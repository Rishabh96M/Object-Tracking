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

cap = cv2.VideoCapture('Videos/ball_video1.mp4')

x = []
y = []

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        red = frame[:, :, 2]
        ret, thresh = cv2.threshold(red, 230, 255, cv2.THRESH_BINARY_INV)
        x_1, y_1, w, h = cv2.boundingRect(thresh)
        row, col = np.where(thresh == 255)

        top = [col[0], row[0]]
        bottom = [col[-1], row[-1]]

        x.append(top[0])
        y.append(top[1])
        x.append(bottom[0])
        y.append(bottom[1])

        cv2.circle(frame, top, 8, (0, 255, 0), -1)
        cv2.circle(frame, bottom, 8, (0, 255, 0), -1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

plt.plot(x, y, 'ro')

y_new = estimations.ols(x, y, 2)

plt.plot(x, y_new, 'g-')
plt.gca().invert_yaxis()
plt.show()
