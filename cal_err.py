import numpy as np
import os
import cv2
from util import readPFM, cal_avgerr


def main():
    print('Compute error')
    N = 10
    imgae_dir = './data/Synthetic'
    output_dir = './output/Synthetic'
    err_list = []
    for i in range(N):
        print("loading image %d/10" % i)
        gt = readPFM(os.path.join(imgae_dir, "TLD%d.pfm" % (i) ))#os.path.join(imgae_dir, "%04d_0.png" % (i) )
        disp = readPFM(os.path.join(output_dir, "SD%d.pfm" % (i) ))

        err = cal_avgerr(gt, disp)#GT,disp
        print('avgerr: %f ' % err)
        err_list.append(err)
      
    print('Average error on synthetic data: %.2f' % (sum(err_list)/len(err_list) ))

if __name__ == "__main__":
    main()