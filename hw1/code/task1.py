import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
image_path = './data/1.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Unable to load the image: {image_path}")

# 2. Initialize the SIFT detector
sift = cv2.SIFT_create()

# 3. Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Output the number of keypoints and the shape of the descriptors
print(f"Number of keypoints detected: {len(keypoints)}")
print(f"Shape of descriptors: {descriptors.shape}")

# 4. Draw the keypoints
image_with_keypoints = cv2.drawKeypoints(
    image, 
    keypoints, 
    None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 5. Create a white separator
height = max(image.shape[0], image_with_keypoints.shape[0])
separator_width = 30  # Width of the white padding
separator = 255 * np.ones((height, separator_width, 3), dtype=np.uint8)

# 6. Concatenate the original image, separator, and result side by side
original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
combined_image = cv2.hconcat([original_bgr, separator, image_with_keypoints])

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image (Left) vs SIFT Keypoints (Right)")
plt.axis('off')
plt.tight_layout()
plt.savefig('./results/1.png', dpi=100)
plt.show()
