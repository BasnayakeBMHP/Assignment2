import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt

image = data.camera()

# Define region growing function
def region_growing(img, seed, thresh):
    # Initialize the output image
    height, width = img.shape[:2]
    region = np.zeros_like(img)
    
    # Define the neighbors
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    
    # Create a list to hold the pixels to be checked
    check_list = [seed]
    
    # Loop until all pixels in the list have been checked
    while len(check_list) > 0:
        # Pop the first pixel from the list
        current_point = check_list.pop(0)
        
        # Check the neighbors of the current pixel
        for i in range(8):
            # Calculate the coordinates of the neighbor pixel
            x = current_point[0] + neighbors[i][0]
            y = current_point[1] + neighbors[i][1]
            
            # Check if the neighbor is within the image boundaries
            if x >= 0 and y >= 0 and x < height and y < width:
                # Check if the neighbor is within the threshold range
                if abs(img[x, y] - img[current_point]) < thresh and region[x, y] == 0:
                    # Add the neighbor pixel to the region and the check list
                    region[x, y] = 255
                    check_list.append((x, y))
                    
    return region


# Set the seed point and threshold value
seed_point = (100, 100)
threshold = 220


# Apply region growing
region = region_growing(image, seed_point, threshold)

#Display the output image
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(region, cmap='gray')
plt.title('Implemented region-growing')


plt.show()
