import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture('Videos/ball_video2.mp4')

x_top = []
y_top = []
x_bottom = []
y_bottom = []

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        red = frame[:, :, 2]
        ret, thresh = cv2.threshold(red, 240, 255, cv2.THRESH_BINARY_INV)
        x, y, w, h = cv2.boundingRect(thresh)

        top = (int(x + w/2), y)
        bottom = (int(x + w/2), y + h)

        x_top.append(int(x + w/2))
        y_top.append(y)
        x_bottom.append(int(x + w/2))
        y_bottom.append(y + h)
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

X_top = np.array(x_top)
A_top = np.vstack([X_top, np.ones(len(X_top))]).T
m, c = np.linalg.lstsq(A_top, y_top, rcond=None)[0]
plt.plot(x_top, m * X_top + c, 'r')

plt.show()
