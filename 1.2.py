import cv2
import numpy as np
import matplotlib.pyplot as plt
# from skimage.feature import hog
from skimage import exposure
import mahotas
image_path = r'C:\Users\vicky\OneDrive\Desktop\clg\Labs\7th sem\IPCV\image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (64, 128))
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0,ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1,ksize=3)
print("The gradients are : ", gradient_x, gradient_y)
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
orientation = np.arctan2(gradient_y, gradient_x) * (180 / np.pi) 
print("Magintude: ", magnitude)
print("Gradient: ",orientation)

from skimage.feature import hog
features, hog_image = hog(image,visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.title('HOG Image')
plt.axis('off')
plt.show()


print(features)
# Print the HOG feature vector length
print("HOG feature vector length:", len(features))
