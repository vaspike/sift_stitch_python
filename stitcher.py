import cv2
import numpy as np

MIN_MATCH_COUNT = 10
detector = cv2.SIFT_create()
matcher = cv2.BFMatcher


def stitch(img0, img1):
    kp0, des0 = detector.detectAndCompute(img0, None)
    kp1, des1 = detector.detectAndCompute(img1, None)


# 获取匹配点
def get_matches(des1, des2, flag: bool = True) -> []:
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if flag:
        if len(good_matches) > MIN_MATCH_COUNT:
            return good_matches
        else:
            return None
    else:
        return good_matches
