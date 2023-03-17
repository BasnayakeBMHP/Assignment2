import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create an image with 2 objects and a total of 3-pixel values
img = np.zeros((100, 100), dtype=np.uint8)
img[20:60, 20:60] = 1
img[50:90, 50:90] = 2

# Add Gaussian noise to the image
gauss_noise=np.zeros((100,100),dtype=np.uint8)
cv2.randn(gauss_noise,0.6,0.6)
gauss_noise=(gauss_noise*0.5).astype(np.uint8)

#noise = np.random.normal(0, 1, img.shape)
noisy_img =  gauss_noise +  img

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(noisy_img,(5,5),0)
threshold_value, thresholded_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Display the original image, noisy image, and the thresholded image

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(noisy_img, cmap='gray')
plt.title('Image with Noise')

plt.subplot(1,3,3)
plt.imshow(thresholded_img, cmap='gray')
plt.title('Thresholded Image')

plt.show()