import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(img, seed, threshold):
    """
    Perform region growing segmentation.
    
    Parameters:
    - img: Input grayscale image
    - seed: Seed point (x, y)
    - threshold: Intensity difference threshold for region growing
    
    Returns:
    - segmented_image: Binary image with segmented region
    """
    # Initialize the output segmented image
    segmented_image = np.zeros_like(img)
    
    # Get image dimensions
    rows, cols = img.shape
    
    # Create a queue to hold the pixels to be processed
    queue = [seed]
    
    # Get the intensity value at the seed point
    seed_value = img[seed]
    
    # Mark the seed point as visited in the segmented image
    segmented_image[seed] = 255
    
    # 8-connected neighborhood (up, down, left, right, and diagonal)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while queue:
        # Pop the first element in the queue
        x, y = queue.pop(0)
        
        # Check the 8 neighbors
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and segmented_image[nx, ny] == 0:
                # If the pixel intensity is within the threshold, add it to the region
                if abs(int(img[nx, ny]) - int(seed_value)) <= threshold:
                    queue.append((nx, ny))  # Add to the queue
                    segmented_image[nx, ny] = 255  # Mark as part of the region
    
    return segmented_image

# Load the image (grayscale)
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Define the seed point and threshold
seed_point = (50, 50)  # Example seed point (change as needed)
threshold_value = 30  # Intensity threshold for region growing

# Apply region growing segmentation
segmented_image = region_growing(image, seed_point, threshold_value)

# Plot the original and segmented images
plt.figure(figsize=(10, 10))

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Show segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image (Region Growing)')
plt.axis('off')

plt.show()
