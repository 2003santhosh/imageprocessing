import cv2
import os

# Video path
video_path = 'path/to/your/video.mp4'  # Replace with the actual video path
output_folder = 'extracted_images'  # Folder to store extracted images

# Create a folder to store images if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Total frames in the video: {total_frames}')

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if we reach the end of the video

    # Save every frame as an image
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    
    print(f"Extracting frame {frame_count + 1}/{total_frames}")
    frame_count += 1

# Release the video capture object
cap.release()

print("Image extraction completed!")
