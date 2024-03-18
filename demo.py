import cv2
import numpy as np


# 读取待拼接的图像
img1 = cv2.imread('IMG_1786-2.jpg')
img2 = cv2.imread('IMG_1786-2.jpg')
# 初始化SIFT检测器
sift = cv2.SIFT_create()
# sift1 = pysift
# 提取特征点并生成描述符
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# keypoints1, descriptors1 = sift1.computeKeypointsAndDescriptors(img1)
# keypoints2, descriptors2 = sift1.computeKeypointsAndDescriptors(img2)
# 特征匹配
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
# 去除错误匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
# 估计变换矩阵
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# 图像变换和拼接
h, w = img1.shape[:2]
result = cv2.warpPerspective(img1, M, (w, h))
result[0:img2.shape[0], 0:img2.shape[1]] = img2
# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()