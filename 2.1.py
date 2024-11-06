import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = r'image.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1) Contrast Stretching
# Find the minimum and maximum pixel values
min_val = np.min(image)
max_val = np.max(image)

# Apply contrast stretching
stretched_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 2) Linear Filtering (Gaussian filter)
filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

# Show the original image and its histogram
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.hist(image.ravel(), bins=256, range=(0, 255))
plt.title('Histogram of Original Image')
plt.show()

# Show the contrast stretched image and its histogram
plt.figure(figsize=(6, 6))
plt.imshow(stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.hist(stretched_image.ravel(), bins=256, range=(0, 255))
plt.title('Histogram of Contrast Stretched Image')
plt.show()

# Show the filtered image and its histogram
plt.figure(figsize=(6, 6))
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Gaussian Blur)')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.hist(filtered_image.ravel(), bins=256, range=(0, 255))
plt.title('Histogram of Filtered Image')
plt.show()