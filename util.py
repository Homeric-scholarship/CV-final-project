import re
import sys
import numpy as np

def readPFM(file):
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8')
        if 'PF' in header:
            color = True
        elif 'Pf' in header:
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        
        scale = float(f.readline().rstrip())
        if scale < 0:
            endian = '<'    # little-endian
            scale = -scale
        else:
            endian = '>'    # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        
    return data

def writePFM(file, image, scale=1):
    with open(file, 'wb') as f:
        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):  # grayscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        f.write(b'PF\n' if color else b'Pf\n')
        f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
            scale = -scale

        f.write(b'%f\n' % scale)

        image.tofile(f)
    
def cal_avgerr(GT, disp):
    return np.mean(np.abs(GT - disp)[GT != np.inf])