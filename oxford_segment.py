from multiprocessing import Pool
import os
import os.path

import numpy as np

from pixseg import get_pix_labels

IMAGE_DIR = '/home/emartin/186_data/oxford/images/'
SAVE_DIR = '/home/emartin/186_data/oxford/segmented/'

def f(filename):
    data = get_pix_labels(filename)

    fn = '.'.join(os.path.basename(filename).split('.')[:-1])
    full_name = os.path.join(SAVE_DIR, fn + '.npy')
    np.save(full_name, data)
    print 'saved %s' % full_name

p = Pool(4)

_, _, filenames = os.walk(IMAGE_DIR).next()
filenames = [os.path.join(IMAGE_DIR, fn) for fn in filenames]

p.map(f, filenames)
