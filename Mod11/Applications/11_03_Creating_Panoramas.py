import cv2
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'


image_file1 = "./scene/scene1.jpg"  # Reference image.
image_file2 = "./scene/scene3.jpg"  # Image to be aligned.

# Read the images.
img1 = cv2.imread(image_file1, cv2.IMREAD_COLOR)
img2 = cv2.imread(image_file2, cv2.IMREAD_COLOR)

# Convert images to grayscale.
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
MAX_FEATURES = 500 
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

# Draw the keypoints.
img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None, 
	color = (0,0,255),flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None, 
	color = (0,0,255),flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Display.
cv2.imshow('Keypoints for Image-1', img1_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Keypoints for Image-2', img2_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find Matching Corresponding Points.
# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches = sorted(matches, key = lambda x: x.distance, reverse = False)

# Retain only a percenatge of the better matches.
GOOD_MATCH_PERCENT = 0.15
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)

# Draw the Matches.
cv2.imshow('Matches Obtained from the Descriptor Matcher', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

zoom_x1 = 300; zoom_x2 = 1300
zoom_y1 = 300; zoom_y2 = 700

img_matches_zoom = img_matches[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

cv2.imshow('Matches Obtained from the Descriptor Matcher', img_matches_zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Alignment using Homography.
# Compute Homography.
points1 = np.zeros((len(matches), 2), dtype = np.float32)
points2 = np.zeros((len(matches), 2), dtype = np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography.
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# Warp time the perspective of the 2nd image using the homography.
img1_h, img1_w, channels = img1.shape
img2_h, img2_w, channels = img2.shape

img2_aligned = cv2.warpPerspective(img2, h, (img2_w + img1_w, img2_h))

cv2.imshow('Second image aligned to first image obtained using homography and warping', img2_aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Stitch Image-1 with aligned image-2.
stitched_image = np.copy(img2_aligned)
stitched_image[0:img1_h, 0:img1_w] = img1

cv2.imshow('Final Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Alternative method using OpenCV stitcher class.
# Read images.
imagefiles = glob.glob('./scene/*')
imagefiles.sort()
print(imagefiles)

images = []
for filename in imagefiles:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

num_images = len(images)

# Display images.
plt.figure(figsize = [20,10]) 
num_cols = 4
num_rows = math.ceil(num_images / num_cols)

for i in range(0, num_images):
    plt.subplot(num_rows, num_cols, i+1) 
    plt.axis('off')
    plt.imshow(images[i])
# Using stitcher class.
stitcher = cv2.Stitcher_create()
status, panorama = stitcher.stitch(images)
if status == 0:
    plt.figure(figsize = [20,10]) 
    plt.imshow(panorama)
# Press q to exit.
# Crop the panorama.
plt.figure(figsize = [20,10]) 
plt.imshow(panorama)
plt.show()
cropped_region = panorama[90:867, 1:2000]
plt.imshow(cropped_region)
plt.show()