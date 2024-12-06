import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images to be stitched
image1_path = './data/1.png'
image2_path = './data/1_2.png'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)


# Convert images to grayscale for feature detection
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect features and compute descriptors using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match features using FLANN-based matcher
index_params = dict(algorithm=1, trees=5)  # Using KD-Tree
search_params = dict(checks=50)  # Number of times the tree is recursively searched
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract the coordinates of matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate the homography matrix using RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

height, width = image2.shape[:2]
height1, width1 = image1.shape[:2]

result = cv2.warpPerspective(image1, H, (width, height+height1))

# Place the second image in the result
result[0:height, 0:width] = image2

cv2.rectangle(result, (0, 0), (width, height), (0, 255, 0), 5)

# Apply homography to the corners of image1 to draw the transformed rectangle
pts = np.float32([[0, 0], [image1.shape[1], 0], [image1.shape[1], image1.shape[0]], [0, image1.shape[0]]]).reshape(-1, 1, 2)
pts_transformed = cv2.perspectiveTransform(pts, H)

# Draw the transformed rectangle around image1 in the stitched result
pts_transformed = np.int32(pts_transformed)
cv2.polylines(result, [pts_transformed], isClosed=True, color=(255, 0, 0), thickness=5)

plt.figure(figsize=(6, 12))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Image Stitching with Homography (Blue: image1, Green: image2)")
plt.axis('off')
plt.tight_layout()
plt.savefig('./results/3.png', dpi=100)
plt.show()