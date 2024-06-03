import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'star_sky.jpg' 
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, threshold_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = np.zeros_like(image)

for contour in contours:
    cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Detected Celestial Bodies')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

plt.show()