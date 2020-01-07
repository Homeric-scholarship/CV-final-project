import numpy as np
import cv2
import time
import sys
import os
from util import readPFM, writePFM, cal_avgerr

test = 1 # 0: Synthetic 1:Real
if test == 0:
    tic = time.time()
    N = 10
    imgae_dir = './data/Synthetic'
    for i in range(N):
        print("loading image %d/10" % i)
        Lpath = os.path.join(imgae_dir, "TL%d.png" % (i) )#os.path.join(imgae_dir, "%04d_0.png" % (i) )
        Rpath = os.path.join(imgae_dir, "TR%d.png" % (i) ) 
        outputpath = os.path.join(imgae_dir, "SD%d.pfm" % (i) ) 

        cmd = 'python3 main.py --input-left '+ Lpath +' --input-right '+ Rpath +' --output '+ outputpath
        os.system(cmd)

    toc = time.time()
    print('All time: %f sec.' % (toc - tic))
elif test == 1:
    tic = time.time()
    N = 10
    imgae_dir = './data/Real'
    for i in range(N):
        print("loading image %d/10" % i)
        Lpath = os.path.join(imgae_dir, "TL%d.bmp" % (i) )#os.path.join(imgae_dir, "%04d_0.png" % (i) )
        Rpath = os.path.join(imgae_dir, "TR%d.bmp" % (i) ) 
        outputpath = os.path.join(imgae_dir, "RD%d.pfm" % (i) ) 

        cmd = 'python3 main.py --input-left '+ Lpath +' --input-right '+ Rpath +' --output '+ outputpath
        os.system(cmd)

    toc = time.time()
    print('All time: %f sec.' % (toc - tic))