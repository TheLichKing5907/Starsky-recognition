import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'star_sky.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, threshold_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

star_threshold_area = 500

result_image = image.copy()

for contour in contours:
    area = cv2.contourArea(contour)
    if area < star_threshold_area:

        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(result_image, center, radius, (0, 255, 0), 2)
    else:

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Detected and Classified Celestial Bodies')
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

plt.show()
