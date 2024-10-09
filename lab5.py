# Task 5
import cv2

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Display the grayscale frame
    cv2.imshow('Grayscale Frame', gray_frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

















#TAsk 1 part a
# height, width, channels = image.shape
# print(f"Dimensions: {width}x{height}")
# print(f"Number of Channels: {channels}")
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display
# plt.imshow(image_rgb)
# plt.axis('off')  # Turn off axis labels
# plt.show()



#Task 1 part b
#from PIL import Image
# import os
# import time
# image_path = 'parrot.jpg'  
# image = Image.open(image_path)
# # Get the size (in bytes)
# image_size = os.path.getsize(image_path)

# # Get the creation or modification date
# modification_time = os.path.getmtime(image_path)
# modification_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modification_time))

# # Get width, height, and bit depth
# width, height = image.size
# bit_depth = image.mode  # Example: 'RGB' or 'L' (grayscale)

# # Get coding method (compression method, if applicable)
# image_format = image.format  # Example: 'JPEG'

# # Display all the extracted information
# print(f"File name: {os.path.basename(image_path)}")
# print(f"Size: {image_size / 1024:.2f} KB")
# print(f"Date modified: {modification_date}")
# print(f"Coding method: {image_format}")
# print(f"Bit depth (Mode): {bit_depth}")
# print(f"Width: {width}px")
# print(f"Height: {height}px")



#Task 1 part c

# import cv2
# import numpy as np

# # Load the image as double (float64 type) to introduce out-of-range values
# image_path = 'parrot.jpg'  
# image = cv2.imread(image_path).astype(np.float64)

# # Modify the image by introducing out-of-range values
# image_out_of_range = image * 2  # This will create values beyond [0, 255]

# # Convert the out-of-range float64 image to uint8
# image_uint8 = image_out_of_range.astype(np.uint8)

# # Check the original and the converted image shapes and type
# print("Original shape and type:", image_out_of_range.shape, image_out_of_range.dtype)
# print("Converted shape and type:", image_uint8.shape, image_uint8.dtype)

# # Check minimum and maximum values before and after conversion
# print("Min/Max values in original image:", image_out_of_range.min(), image_out_of_range.max())
# print("Min/Max values in converted image:", image_uint8.min(), image_uint8.max())



#Task 2
# import cv2
# import matplotlib.pyplot as plt

# # Load the image
# image_path = 'parrot.jpg'  
# image = cv2.imread(image_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Binarize the image using cv2.threshold with a threshold of 127
# _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# # Display the original and binarized images
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Binarized Image')
# plt.imshow(binary_image, cmap='gray')
# plt.axis('off')

# plt.show()



# Task 3
# import cv2
# import matplotlib.pyplot as plt

# # Load the image
# image_path = 'parrot.jpg'  
# image = cv2.imread(image_path)

# height, width, channels = image.shape 
#     # Display the size and number of channels
# if channels == 3:
#         print("This is a three-channel image.")
        
#         # Convert to grayscale
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Binarize the gray image
#         _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
#         # Display the original and binarized images
#         plt.figure(figsize=(12, 6))

#         plt.subplot(1, 3, 1)
#         plt.title('Original Image')
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.title('Grayscale Image')
#         plt.imshow(gray_image, cmap='gray')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.title('Binarized Image')
#         plt.imshow(binary_image, cmap='gray')
#         plt.axis('off')

#         plt.show()

# else:
#         print("This is a one-channel image.")
        
#         # Just display the one-channel image
#         plt.figure(figsize=(6, 6))
#         plt.title('One-Channel Image')
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')
#         plt.show()



# Task 4
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = 'parrot.jpg' 
# image = cv2.imread(image_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Define the threshold
# threshold = 127

# # Get the dimensions of the image
# height, width = gray_image.shape

# # Create a binary output image
# binary_image = np.zeros((height, width), dtype=np.uint8)

# # Binarize the image using nested loops
# for i in range(height):
#     for j in range(width):
#         if gray_image[i, j] >= threshold:
#             binary_image[i, j] = 255  # White
#         else:
#             binary_image[i, j] = 0    # Black

# # Display the original and binarized images
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Binarized Image')
# plt.imshow(binary_image, cmap='gray')
# plt.axis('off')

# plt.show()

