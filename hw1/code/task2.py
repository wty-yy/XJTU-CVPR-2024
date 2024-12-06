import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load two images
image1_path = './data/1.png'
image2_path = './data/1_2.png'
image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None:
    raise FileNotFoundError("One or both images cannot be loaded.")

# 2. Initialize SIFT detector
sift = cv2.SIFT_create()

# 3. Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 4. Match descriptors using FLANN matcher
flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 5. Apply Lowe's ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # Ratio test threshold
        good_matches.append(m)

print(f"Number of good matches after ratio test: {len(good_matches)}")

# 6. Extract matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 7. Use RANSAC to find the homography matrix
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

print(f"Number of inliers after RANSAC: {np.sum(matches_mask)}")

# 8. Draw matches with inliers
draw_params = dict(matchColor=(0, 255, 0),  # Green matches (inliers)
                   singlePointColor=(255, 0, 0),  # Blue keypoints
                   matchesMask=matches_mask,  # Show inliers only
                   flags=cv2.DrawMatchesFlags_DEFAULT)

result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, **draw_params)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Feature Matching with RANSAC")
plt.axis('off')
plt.tight_layout()
plt.savefig("./results/2.png", dpi=100)
plt.show()
