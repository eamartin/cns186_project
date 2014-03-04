import csv
import os
import os.path
import shutil
import subprocess
import sys

import numpy as np
from scipy.misc import imread, imresize, imsave

PIXSEG_DIR = 'pixseg/'
EXECUTABLE = os.path.join(PIXSEG_DIR, 'inferPixelLabels')
CONFIG = os.path.join(PIXSEG_DIR, 'config.xml')
MODEL = os.path.join(PIXSEG_DIR, 'pixelSegModel.xml')
SETUP = 'LD_LIBRARY_PATH=~/186_data/darwin/trunk/external/opencv/lib'

INPUT_FILE = os.path.join(PIXSEG_DIR, 'in.txt')
OUTPUT_SUFFIX = '.out.txt'

def get_pix_labels(image_file, delete=True):
    file_name = os.path.basename(image_file)
    temp_image_path = os.path.join(PIXSEG_DIR, file_name)

    # resize image
    image = imread(os.path.expanduser(image_file))
    original_size = image.shape[:-1]
    smaller = imresize(image, 300.0 / max(image.shape))
    imsave(temp_image_path, smaller)

    #shutil.copy(image_file, temp_image_path)

    with open(INPUT_FILE, 'w') as f:
        f.write(file_name + '\n')

    command = ('%s %s -config %s -outLabels %s %s' %
               (SETUP, EXECUTABLE, CONFIG, OUTPUT_SUFFIX, INPUT_FILE))

    with open('/dev/null') as devnull:
        subprocess.check_call(command, shell=True,
                              stdout=devnull, stderr=devnull)

    with open(temp_image_path + OUTPUT_SUFFIX) as f:
        lines = f.readlines()

    lines = [map(int, l.split()) for l in lines]
    data = imresize(np.array(lines, dtype=np.uint8),
                    original_size, interp='nearest')

    os.remove(temp_image_path)
    os.remove(INPUT_FILE)

    if delete:
        os.remove(temp_image_path + OUTPUT_SUFFIX)

    return data


def main():
    get_pix_labels(sys.argv[1], delete=False)

if __name__ == '__main__':
    main()
