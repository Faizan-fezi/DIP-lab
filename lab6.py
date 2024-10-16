#LAb 6 all tasks upside down
# #Task 4
from PIL import Image, ImageChops
import numpy as np

import cv2

# Load the images
img1_path = 'Picture1.png'
img2_path = 'Picture2.png'
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# Convert images to grayscale
img1_gray = img1.convert('L')
img2_gray = img2.convert('L')

# Perform image difference operation
diff = ImageChops.difference(img1_gray, img2_gray)

# Convert difference image to numpy array for analysis
diff_np = np.array(diff)

# Highlight significant differences by thresholding
threshold = 30  # set a threshold for highlighting changes
diff_np_highlighted = np.where(diff_np > threshold, 255, 0).astype(np.uint8)

# Create an Image from highlighted difference
diff_highlighted_img = Image.fromarray(diff_np_highlighted)

# Save the highlighted difference image to inspect which areas have motion
diff_highlighted_img.show()







#Task 3
# import cv2
# import numpy as np

# # Step 1: Load foreground image (green screen image) and background images
# foreground_image = cv2.imread('greenbackground.jpg')
# background_image1 = cv2.imread('denys-nevozhai-HhmCIJTLuGY-unsplash.jpg')
# background_image2 = cv2.imread('hendrik-morkel-mP7OJFMarfI-unsplash.jpg')
# background_image3 = cv2.imread('nick-fewings-Pw_MSr7kvVg-unsplash.jpg')

# foreground_image = cv2.resize(foreground_image,(400,500))
# # Step 2: Resize the background images to match the foreground size
# background_image1 = cv2.resize(background_image1, (foreground_image.shape[1], foreground_image.shape[0]))
# background_image2 = cv2.resize(background_image2, (foreground_image.shape[1], foreground_image.shape[0]))
# background_image3 = cv2.resize(background_image3, (foreground_image.shape[1], foreground_image.shape[0]))

# # Step 3: Convert the foreground image from BGR to HSV
# hsv_foreground = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2HSV)

# # Step 4: Define the HSV range for green
# lower_green = np.array([35, 100, 100])
# upper_green = np.array([85, 255, 255])

# # Step 5: Create a mask to detect the green color
# mask = cv2.inRange(hsv_foreground, lower_green, upper_green)

# # Step 6: Invert the mask to get the object (non-green areas)
# mask_inv = cv2.bitwise_not(mask)

# # Step 7: Extract the object (foreground) without the green background
# fg_part = cv2.bitwise_and(foreground_image, foreground_image, mask=mask_inv)

# # Function to blend with a background image
# def apply_background(background_image):
#     # Step 8: Extract the corresponding region from the background image
#     bg_part = cv2.bitwise_and(background_image, background_image, mask=mask)

#     # Step 9: Combine the foreground and background
#     final_image = cv2.add(fg_part, bg_part)

#     return final_image

# # Apply the green screen effect with three different backgrounds
# final_image1 = apply_background(background_image1)
# final_image2 = apply_background(background_image2)
# final_image3 = apply_background(background_image3)

# cv2.imshow('Original Green Screen Image', foreground_image)
# cv2.imshow('Background 1 Applied', final_image1)
# cv2.imshow('Background 2 Applied', final_image2)
# cv2.imshow('Background 3 Applied', final_image3)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow('Original Green Screen Image', cv2.resize(foreground_image,(400,250)))
# cv2.imshow('Background 1 Applied', cv2.resize(final_image1,(400,250)))
# cv2.imshow('Background 2 Applied', cv2.resize(final_image2,(400,250)))
# cv2.imshow('Background 3 Applied', cv2.resize(final_image3,(400,250)))


#Task 2
# import cv2
# import numpy as np

# image = cv2.imread('parrot.jpg')

# # Define a scalar for operations
# scalar_value = 50  # You can adjust this value

# # (a) Addition: Increases pixel values, brightening the image.
# addition_image = cv2.resize(cv2.add(image, np.full(image.shape, scalar_value, dtype=np.uint8)),(300,200))

# # (b) Subtraction: Decreases pixel values, darkening the image.
# subtraction_image = cv2.resize(cv2.subtract(image, np.full(image.shape, scalar_value, dtype=np.uint8)),(300,200))

# # (c) Multiplication: Amplifies pixel values, enhancing brightness and contrast.
# # Convert the image to float32 for multiplication
# image_float = image.astype(np.float32)
# multiplication_image = cv2.multiply(image_float, 2.2)
# multiplication_image = cv2.resize(np.clip(multiplication_image, 0, 255).astype(np.uint8),(300,200))  # Convert back to uint8

# # (d) Division: Reduces pixel values, making the image dimmer or muted in tone.
# division_image = cv2.divide(image_float, 2.2)
# division_image = cv2.resize(np.clip(division_image, 0, 255).astype(np.uint8),(300,200))  # Convert back to uint8

# cv2.imshow('Original Image', image)
# cv2.imshow('Addition Image', addition_image)
# cv2.imshow('Subtraction Image', subtraction_image)
# cv2.imshow('Multiplication Image', multiplication_image)
# cv2.imshow('Division Image', division_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()




#Task 1d
# import cv2
# import numpy as np

# image = cv2.imread('parrot.jpg')

# negative_image = 255 - image

# cv2.imshow('Original Image', image)
# cv2.imshow('Negative Image', negative_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



#Task 1c
# import cv2
# import numpy as np

# # Function to apply gamma correction
# def adjust_gamma(image, gamma=1.0):
#     # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
#     inv_gamma = 1.0 / gamma
#     table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype(np.uint8)

#     # Apply the gamma correction using the lookup table
#     return cv2.LUT(image, table)

# image = cv2.imread('parrot.jpg')

# # Apply gamma correction (change gamma value for different results)
# gamma = 2.2  
# gamma_corrected_image = adjust_gamma(image, gamma=gamma)

# cv2.imshow('Original Image', image)
# cv2.imshow('Gamma Corrected Image', gamma_corrected_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



#Task 1b
# import cv2
# import numpy as np

# image = cv2.imread('parrot.jpg')

# # Adjust contrast and brightness
# alpha = 1.5  # Contrast control (1.0-3.0)
# beta = 50    # Brightness control (0-100)

# adjusted_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

# cv2.imshow('Original Image', image)
# cv2.imshow('Contrast Adjusted Image', adjusted_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



#Task 1a
# import cv2
# import numpy as np

# image = cv2.imread('parrot.jpg')

# # Adjust brightness
# brightness = 70  # Positive value to increase brightness, negative to decrease
# darkness= -70
# bright_image = cv2.resize(cv2.convertScaleAbs(image, alpha=1, beta=brightness), (400,300))
# dark_image = cv2.resize(cv2.convertScaleAbs(image, alpha=1, beta=darkness), (400,300))


# cv2.imshow('Original Image', cv2.resize(image,(00,300)))
# cv2.imshow('Brightened Image', bright_image)
# cv2.imshow('Darked Image', dark_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# #****************Annotation process*************
# # Convert images to OpenCV format (RGB to BGR for OpenCV)
# img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
# img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

# # Create a binary mask for the difference, focusing on vehicle movements
# _, binary_diff_refined = cv2.threshold(diff_np, threshold, 255, cv2.THRESH_BINARY)

# # Apply morphological operations to clean the mask (remove noise and small differences)
# kernel = np.ones((5,5), np.uint8)
# binary_diff_refined = cv2.morphologyEx(binary_diff_refined, cv2.MORPH_CLOSE, kernel)

# # Find contours in the refined binary difference image
# contours_refined, _ = cv2.findContours(binary_diff_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Annotate only the areas of significant movement
# for contour in contours_refined:
#     if cv2.contourArea(contour) > 500:  # Adjust this threshold to capture only moved vehicles
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(img1_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Highlight in green

# # Convert back to PIL format for display/saving
# annotated_img = Image.fromarray(cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB))
# annotated_img.show()
