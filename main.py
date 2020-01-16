import numpy as np
import argparse
import cv2
import time
from util import writePFM

DEBUG = False

parser = argparse.ArgumentParser(description="Disparity Estimation")
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
    "--output", default="./TL0.pfm", type=str, help="left disparity map"
)

# You can modify the function interface as you like
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
# You can modify the function interface as you like
def computeDisp(Il, Ir, max_disp=64):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)

    # BGR2GRAY
    imgL = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)#.astype(np.float32)
    imgR = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)#.astype(np.float32)

    # histogram equalization
    imgL = cv2.equalizeHist(imgL).astype(np.float32)
    imgR = cv2.equalizeHist(imgR).astype(np.float32)

    # padding with size 32
    block_size = 5
    half_size = int((block_size-1)/2)#2
    padding_size = half_size*16

    imgL = cv2.copyMakeBorder( imgL,padding_size,padding_size,padding_size,padding_size,cv2.BORDER_REFLECT)
    imgR = cv2.copyMakeBorder( imgR,padding_size,padding_size,padding_size,padding_size,cv2.BORDER_REFLECT)#BORDER_REPLICATE


    imgh,imgw = imgL.shape[:2]
    disL = np.zeros((imgh, imgw), dtype=np.float32)
    num_disp = block_size + max_disp # search range
    distmp = np.zeros((imgh, imgw)) # check if the pixel is filled

    for i in range(half_size,imgh-half_size):
        for j in range(half_size,imgw-half_size):
            tpl=imgL[i-half_size:i+half_size+1,j-half_size:j+half_size+1]
            left_bound = j-num_disp
            right_bound = j+half_size+4# negative disparity map
            if left_bound < 0:
                left_bound = 0
            if right_bound >= imgw:
                right_bound = imgw-1
            target=imgR[i-half_size:i+half_size+1,left_bound:right_bound+1]

            result=cv2.matchTemplate(target,tpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            max_loc = (max_loc[0]+left_bound,max_loc[1])


            if distmp[i,max_loc[0]+half_size] == 0 :
                disL[i,j] =  j-(max_loc[0]+half_size)
                distmp[i,max_loc[0]+half_size] =1

    disL = disL[padding_size:-padding_size,padding_size:-padding_size].astype(np.uint8)
    disL = np.where(disL < max_disp+16, disL, 0.1) # adjust negative value to 0.1

    # hole filling
    FL = disL.copy()
    for i in range(disL.shape[0]):
        maybe_valid = 0
        for j in range(disL.shape[1]):#from left to right
            if FL[i,j] != 0:
                maybe_valid = FL[i,j]
            else:
                FL[i,j] = maybe_valid

    # Disparity refinement
    ##weighted median filter
    refined_disparity_map = cv2.ximgproc.weightedMedianFilter(cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY), FL.astype(np.uint8), 21)

    return refined_disparity_map.astype(np.float32)#.astype(np.uint8)


def main():
    DEBUG = True
    args = parser.parse_args()

    print(args.output)
    print("Compute disparity for %s" % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    max_d = max_disp(img_left, img_right)
    disp = computeDisp(img_left, img_right, max_disp=max_d+16)
    toc = time.time()
    writePFM(args.output, disp)
    print("Elapsed time: %f sec." % (toc - tic))


if __name__ == "__main__":
    main()
