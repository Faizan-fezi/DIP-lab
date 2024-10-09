import cv2
import matplotlib.pyplot as plt

image = cv2.imread('parrot.jpg')
height, width, channels = image.shape
print(f"Dimensions: {width}x{height}")
print(f"Number of Channels: {channels}")