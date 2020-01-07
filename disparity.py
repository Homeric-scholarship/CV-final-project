import numpy as np
import time
import cv2
import argparse

DEBUG = False


def max_disp(left_img, right_img):
    """
    get the max disparity
    """

    # Max initial matches
    MAX_INI_MATCH = np.inf

    # ORB detector and brute-force matcher
    detector = cv2.ORB_create(nfeatures=100000, edgeThreshold=5)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Find the keypoints and descriptors of both images
    keypoint_left, descriptor_left = detector.detectAndCompute(
        image=left_img, mask=None
    )
    keypoint_right, descriptor_right = detector.detectAndCompute(
        image=right_img, mask=None
    )

    # Find the best matches
    matches = matcher.match(
        queryDescriptors=descriptor_left, trainDescriptors=descriptor_right
    )
    matches = sorted(matches, key=lambda x: x.distance)

    if DEBUG:
        print(f"Initial matches#")
        print(f"match_lr: {len(matches)}")

    # Match refinement with RANSAC
    left_pts = np.float32([keypoint_left[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    right_pts = np.float32([keypoint_right[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    homography, mask = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    ransac_matches = []
    for (m, keep) in zip(matches, matchesMask):
        if keep:
            ransac_matches.append(m)

    if DEBUG:
        print(f"After RANSAC:")
        print(f"match: {len(ransac_matches)}")
        ransac_img = cv2.drawMatches(
            left_img,
            keypoint_left,
            right_img,
            keypoint_right,
            ransac_matches,
            None,
            flags=2,
        )
        cv2.imwrite("./debug/RANSAC.png", ransac_img)

    # Get the range of disparity
    disp_min = np.inf
    disp_max = 0

    for m in ransac_matches:
        # Keypoints coordinates
        pt_left = np.float32(keypoint_left[m.queryIdx].pt)
        pt_right = np.float32(keypoint_right[m.trainIdx].pt)

        # Compute the distance
        dist = int(abs(pt_left - pt_right)[0])

        if disp_max < dist:
            disp_max = dist

    # Ceiling the number
    disp_max += 1

    return disp_max


def main():
    parser = argparse.ArgumentParser(description="Disparity Range")
    parser.add_argument(
        "--input-left", default="data/Real/TL6.bmp", type=str, help="input left image"
    )
    parser.add_argument(
        "--input-right", default="data/Real/TR6.bmp", type=str, help="input right image"
    )
    args = parser.parse_args()

    print("Compute disparity range %s" % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = max_disp(img_left, img_right)
    toc = time.time()
    print(f"max disparity: {disp}")
    print("Elapsed time: %f sec." % (toc - tic))


if __name__ == "__main__":
    DEBUG = True
    main()
