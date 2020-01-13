import os
import sys
import glob
from tqdm import tqdm

OUTPUT_FOLDER = './output/'
FILE_EXTENSION = '*.pfm'

def main():
    '''
    Turn all the .pfm to .png
    '''
    Filenames = []
    for folder in os.listdir(OUTPUT_FOLDER):
        Filenames.append(sorted(glob.glob(os.path.join(OUTPUT_FOLDER, folder, FILE_EXTENSION))))

    for filenames in Filenames:
        if filenames[0].replace(OUTPUT_FOLDER, '').rsplit('/')[0] == str(sys.argv[1]):
            for filename in tqdm(filenames):
                output_filename = filename.rsplit('.', maxsplit=1)[0] + '.png'
                cmd = 'python3 visualize.py ' + filename + ' ' + output_filename
                os.system(cmd)

if __name__ == "__main__":
    main()