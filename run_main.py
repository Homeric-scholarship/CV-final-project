import os
import sys
import cv2
import time
import numpy as np
from util import *

# 0: Synthetic 1: Real
def main():
    if sys.argv[1] == 'Synthetic':
        tic = time.time()
        N = 10
        imgae_dir = './data/Synthetic/'
        output_dir = './output/Synthetic/'

        for i in range(N):
            print('loading image %d/10' % i)
            Lpath = os.path.join(imgae_dir, 'TL%d.png' % (i))  #os.path.join(imgae_dir, "%04d_0.png" % (i) )
            Rpath = os.path.join(imgae_dir, 'TR%d.png' % (i)) 
            outputpath = os.path.join(output_dir, 'SD%d.pfm' % (i)) 

            cmd = 'python3 main.py --input-left '+ Lpath +' --input-right '+ Rpath +' --output '+ outputpath
            os.system(cmd)

        toc = time.time()
        print('All time: %f sec.' % (toc - tic))
    elif sys.argv[1] == 'Real':
        tic = time.time()
        N = 10
        imgae_dir = './data/Real/'
        output_dir = './output/Real/'

        for i in range(N):
            print('loading image %d/10' % i)
            Lpath = os.path.join(imgae_dir, 'TL%d.bmp' % (i) )#os.path.join(imgae_dir, "%04d_0.png" % (i) )
            Rpath = os.path.join(imgae_dir, 'TR%d.bmp' % (i) ) 
            outputpath = os.path.join(output_dir, 'RD%d.pfm' % (i) ) 

            cmd = 'python3 main.py --input-left '+ Lpath +' --input-right '+ Rpath +' --output '+ outputpath
            os.system(cmd)

        toc = time.time()
        print('All time: %f sec.' % (toc - tic))

if __name__ == "__main__":
    main()