import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r'image.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

# Display original image
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# 1. Scaling
scale_percent = 50  # Scale image to 50% of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
scaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

plt.subplot(2, 2, 2)
plt.imshow(scaled_image)
plt.title("Scaled Image (50%)")
plt.axis('off')

# 2. Rotation
angle = 45  # Rotate the image by 45 degrees
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

plt.subplot(2, 2, 3)
plt.imshow(rotated_image)
plt.title("Rotated Image (45 degrees)")
plt.axis('off')

# 3. Shearing
shear_factor = 0.2
M_shear = np.array([[1, shear_factor, 0],
                    [shear_factor, 1, 0],
                    [0, 0, 1]], dtype=float)

sheared_image = cv2.warpPerspective(image, M_shear, (int(w * (1 + shear_factor)), int(h * (1 + shear_factor))))

plt.subplot(2, 2, 4)
plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')

# Display all images
plt.tight_layout()
plt.show()
