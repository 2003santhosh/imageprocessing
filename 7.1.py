import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detection(image):
    # Step 1: Smoothing the image using Gaussian filter
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
    
    # Step 2: Compute the gradient in both the x and y directions
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel operator in x direction
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel operator in y direction
    
    # Step 3: Calculate the magnitude and angle of the gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)  # Gradient magnitude
    angle = np.arctan2(grad_y, grad_x)  # Gradient direction
    
    # Step 4: Non-Maximum Suppression (thinning of edges)
    non_max_suppressed = non_maximum_suppression(magnitude, angle)
    
    # Step 5: Apply Double Thresholding
    edges = double_threshold(non_max_suppressed)
    
    # Step 6: Edge Tracking by Hysteresis (optional)
    final_edges = edge_tracking_by_hysteresis(edges)
    
    return final_edges

def non_maximum_suppression(magnitude, angle):
    rows, cols = magnitude.shape
    suppressed_image = np.zeros_like(magnitude)
    
    # Quantize the angle to 4 directions: 0, 45, 90, 135 degrees
    angle = np.rad2deg(angle) % 180  # Convert to degrees and ensure it's between 0-180 degrees
    angle[angle < 0] += 180
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # Get the gradient directions and suppress accordingly
            q = 255
            r = 255
            
            # Angle 0 degrees (horizontal edge)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # Angle 45 degrees (diagonal edge)
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # Angle 90 degrees (vertical edge)
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # Angle 135 degrees (diagonal edge)
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed_image[i, j] = magnitude[i, j]
            else:
                suppressed_image[i, j] = 0
    
    return suppressed_image

def double_threshold(image):
    high_threshold = image.max() * 0.2
    low_threshold = high_threshold * 0.5
    
    strong_edges = np.zeros_like(image, dtype=np.uint8)
    weak_edges = np.zeros_like(image, dtype=np.uint8)
    
    # Strong edges: above the high threshold
    strong_edges[image >= high_threshold] = 255
    # Weak edges: between high and low threshold
    weak_edges[(image >= low_threshold) & (image < high_threshold)] = 50
    
    return strong_edges, weak_edges

def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    final_edges = np.copy(strong_edges)
    
    for i in range(1, strong_edges.shape[0]-1):
        for j in range(1, strong_edges.shape[1]-1):
            if strong_edges[i, j] == 255:
                # Check for weak edges connected to strong edges
                if (weak_edges[i + 1, j] == 50) or (weak_edges[i - 1, j] == 50) or \
                   (weak_edges[i, j + 1] == 50) or (weak_edges[i, j - 1] == 50) or \
                   (weak_edges[i + 1, j + 1] == 50) or (weak_edges[i - 1, j - 1] == 50) or \
                   (weak_edges[i + 1, j - 1] == 50) or (weak_edges[i - 1, j + 1] == 50):
                    final_edges[i, j] = 255
                else:
                    final_edges[i, j] = 0
    
    return final_edges

# Example usage
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
edges = canny_edge_detection(image)

# Display the edges
plt.figure(figsize=(10, 10))
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection using Canny')
plt.axis('off')
plt.show()
