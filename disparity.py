import numpy as np
import time
import cv2
import argparse

DEBUG = False


def disp_range(left_img, right_img):
    """
    return (min, max) of the disparity
    """

    # ORB detector and brute-force matcher
    detector = cv2.ORB_create(nfeatures=100000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Find the keypoints and descriptors of both images
    keypoint_left, descriptor_left = detector.detectAndCompute(
        image=left_img, mask=None
    )
    keypoint_right, descriptor_right = detector.detectAndCompute(
        image=right_img, mask=None
    )

    # Find the best 2 matches in both directions
    match_lr = matcher.knnMatch(
        queryDescriptors=descriptor_left, trainDescriptors=descriptor_right, k=2
    )
    match_rl = matcher.knnMatch(
        queryDescriptors=descriptor_right, trainDescriptors=descriptor_left, k=2
    )

    if DEBUG:
        print(f"Initial matches#")
        print(f"match_lr: {len(match_lr)}\t match_rl: {len(match_rl)}")

    # ------- Match Refinement -----
    # 1. Ratio test
    ratio_thresh = 0.9
    ratio_match_lr = []
    ratio_match_rl = []

    for m, n in match_lr:
        if m.distance < ratio_thresh * n.distance:
            ratio_match_lr.append(m)

    for m, n in match_rl:
        if m.distance < ratio_thresh * n.distance:
            ratio_match_rl.append(m)

    if DEBUG:
        print(f"After ratio test:")
        print(f"match_lr: {len(ratio_match_lr)}\t match_rl: {len(ratio_match_rl)}")
        ratio_img = cv2.drawMatches(
            left_img,
            keypoint_left,
            right_img,
            keypoint_right,
            ratio_match_lr,
            None,
            flags=2,
        )
        cv2.imwrite("./debug/ratio_test.png", ratio_img)

    # 2. Symmetry test
    sym_match = []
    # match left -> match right
    for m_lr in ratio_match_lr:
        # match right -> match left
        for m_rl in ratio_match_rl:
            if m_lr.queryIdx == m_rl.trainIdx and m_lr.trainIdx == m_rl.queryIdx:
                sym_match.append(m_lr)
                break  # Next match left -> match right

    if DEBUG:
        print(f"After symmetry test:")
        print(f"match: {len(sym_match)}")
        symm_img = cv2.drawMatches(
            left_img, keypoint_left, right_img, keypoint_right, sym_match, None, flags=2
        )
        cv2.imwrite("./debug/sym_test.png", symm_img)

    # 3. RANSAC
    left_pts = np.float32([keypoint_left[m.queryIdx].pt for m in sym_match]).reshape(
        -1, 1, 2
    )
    right_pts = np.float32([keypoint_right[m.trainIdx].pt for m in sym_match]).reshape(
        -1, 1, 2
    )
    homography, mask = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    if DEBUG:
        print(f"After RANSAC:")
        print(f"match: {len(homography)}")
        draw_params = dict(matchesMask=matchesMask, flags=2)
        ransac_img = cv2.drawMatches(
            left_img,
            keypoint_left,
            right_img,
            keypoint_right,
            sym_match,
            None,
            **draw_params,
        )
        cv2.imwrite("./debug/RANSAC.png", ransac_img)

    disp_min = 0
    disp_max = 0
    return disp_min, disp_max


def main():
    parser = argparse.ArgumentParser(description="Disparity Range")
    parser.add_argument(
        "--input-left",
        default="./data/Synthetic/TL0.png",
        type=str,
        help="input left image",
    )
    parser.add_argument(
        "--input-right",
        default="./data/Synthetic/TR0.png",
        type=str,
        help="input right image",
    )
    parser.add_argument(
        "--output", default="./ORB.png", type=str, help="left disparity map"
    )
    args = parser.parse_args()

    print("Compute disparity range %s" % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = disp_range(img_left, img_right)
    toc = time.time()
    print("Elapsed time: %f sec." % (toc - tic))


if __name__ == "__main__":
    DEBUG = True
    main()
