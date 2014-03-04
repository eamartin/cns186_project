import csv
import os
import os.path
import shutil
import subprocess
import sys

import numpy as np

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
    shutil.copy(image_file, temp_image_path)

    with open(INPUT_FILE, 'w') as f:
        f.write(file_name + '\n')

    command = ('%s %s -config %s -outLabels %s %s' %
               (SETUP, EXECUTABLE, CONFIG, OUTPUT_SUFFIX, INPUT_FILE))
    print command

    with open('/dev/null') as devnull:
        subprocess.check_call(command, shell=True,
                              stdout=devnull, stderr=devnull)

    with open(temp_image_path + OUTPUT_SUFFIX) as f:
        reader = csv.reader(f, delimiter=' ')
        lines = list(reader)

    lines = [map(int, l) for l in lines]
    data = np.array(lines)
    print data

    os.remove(temp_image_path)
    os.remove(INPUT_FILE)

    if delete:
        os.remove(temp_image_path + OUTPUT_SUFFIX)

    return data


def main():
    get_pix_labels(sys.argv[1], delete=False)

if __name__ == '__main__':
    main()
