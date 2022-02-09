import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture('Videos/ball_video1.mp4')

x_top = []
y_top = []
x_bottom = []
y_bottom = []

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        red = frame[:, :, 2]
        ret, thresh = cv2.threshold(red, 220, 255, cv2.THRESH_BINARY_INV)
        x, y, w, h = cv2.boundingRect(thresh)
        row, col = np.where(thresh == 255)

        top = [col[0], row[0]]
        bottom = [col[-1], row[-1]]

        x_top.append(top[0])
        y_top.append(top[1])
        x_bottom.append(bottom[0])
        y_bottom.append(bottom[1])

        cv2.circle(frame, top, 8, (0, 255, 0), -1)
        cv2.circle(frame, bottom, 8, (0, 255, 0), -1)

        cv2.imshow('frame', frame)
        cv2.imshow('after thresholding', thresh)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

plt.plot(x_top, y_top, 'ro')
plt.plot(x_bottom, y_bottom, 'bo')

plt.show()
