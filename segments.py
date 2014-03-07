import collections
import os

import numpy as np
from scipy import ndimage
from scipy.misc import imread, imresize, imsave

CLASSES_I_CARE_ABOUT = [0, 2, 7, 12, 15, 16, 20]

def get_segment_masks(seg_labels):
    seg_labels += 1
    masks = []
    for label in CLASSES_I_CARE_ABOUT:
        label += 1
        features, num_features = ndimage.measurements.label(seg_labels == label)
        for i in xrange(1, num_features + 1):
            mask = (features == i)
            size = mask.sum()
            if size > (mask.size / 25):
                masks.append(label * mask)
    return masks

def save_masks():
    IMG_DIR = '/home/emartin/186_data/oxford/images/'
    SEG_DIR = '/home/emartin/186_data/oxford/segmented/'
    SAVE_DIR = '/home/emartin/186_data/oxford/partials/'

    filenames = [fn.split('.')[0] for fn in os.listdir(IMG_DIR)]

    for fn in filenames:
        segmented = np.load(os.path.join(SEG_DIR, '.'.join([fn, 'npy'])))
        masks = get_segment_masks(segmented)

        c = collections.Counter()
        for mask in masks:
            error = False
            label = mask.max()

            if label == 0:
                print 'label 0 for %s' % fn
                error = True
            c[label] += 1

            slices = ndimage.find_objects(mask)

            if slices[label - 1] is None:
                print "find_object didn't find slice"
                error = True
                continue

            image = imread(os.path.join(IMG_DIR, '.'.join([fn, 'jpg'])))
            sliced = image[slices[label - 1]]

            final_filename = '%s_%d_%d' % (fn, label, c[label])
            imsave(os.path.join(SAVE_DIR, '.'.join([final_filename, 'jpg'])),
                   sliced)

            print 'Saved slice %s, error=%s' % (final_filename, error)

if __name__ == '__main__':
    save_masks()
