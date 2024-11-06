import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = 'image.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Constructing a Scale Space and Detecting Keypoints
sift = cv2.SIFT_create()
keypoints = sift.detect(gray_image, None)

# Step 2: Keypoint Localization (part of SIFT's internal process)
# Keypoints are already localized during the `detect` method

# Step 3 & 4: Orientation Assignment and Keypoint Descriptor Computation
keypoints, descriptors = sift.compute(gray_image, keypoints)

# Draw the keypoints on the original image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("Image with SIFT Keypoints")
plt.axis('off')
plt.show()

# Print out some information about keypoints and descriptors
print(f"Number of keypoints detected: {len(keypoints)}")
print("Descriptor shape:", descriptors.shape)
