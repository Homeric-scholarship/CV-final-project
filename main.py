import numpy as np
import argparse
import cv2
import time
from scipy import signal
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
def computeDisp(Il, Ir, max_disp=64):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # padding with size 32
    block_size = 5
    half_size = int((block_size-1)/2)#2
    padding_size = half_size*16
    Il_padding = cv2.copyMakeBorder( Il,padding_size,padding_size,padding_size,padding_size,cv2.BORDER_REFLECT)
    Ir_padding = cv2.copyMakeBorder( Ir,padding_size,padding_size,padding_size,padding_size,cv2.BORDER_REFLECT)#BORDER_REPLICATE

    # BGR2GRAY
    imgL = cv2.cvtColor(Il_padding, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(Ir_padding, cv2.COLOR_BGR2GRAY)

    imgh,imgw = imgL.shape[:2]
    disL = np.zeros((imgh, imgw), dtype=np.float32)
    num_disp = block_size + max_disp # search range
    distmp = np.zeros((imgh, imgw)) # check if the pixel is filled

    for i in range(half_size,imgh-half_size):
        for j in range(half_size,imgw-half_size):
            tpl=imgL[i-half_size:i+half_size+1,j-half_size:j+half_size+1]
            left_bound = j-num_disp
            right_bound = j+half_size
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

    disL = disL[padding_size:-padding_size,padding_size:-padding_size]
    disL = disL.astype(np.uint8)#.astype(np.int)

    # Disparity refinement
    # hole filling
    FL = disL.copy()
    for i in range(disL.shape[0]):
        maybe_valid = 0
        for j in range(disL.shape[1]):#from left to right
            if FL[i,j] != 0:
                maybe_valid = FL[i,j]
            else:
                FL[i,j] = maybe_valid
    # median filter
    labels = signal.medfilt2d(FL,15)


    return labels.astype(np.float32)#.astype(np.uint8)


def main():
    DEBUG = True
    args = parser.parse_args()

    print(args.output)
    print("Compute disparity for %s" % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(img_left, img_right)
    toc = time.time()
    writePFM(args.output, disp)
    print("Elapsed time: %f sec." % (toc - tic))


if __name__ == "__main__":
    main()
