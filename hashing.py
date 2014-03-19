import collections
import cPickle as pickle
import os

import numpy as np

from nnet import OxfordImages, OX_DIR, IMAGE_DIR

TRAIN_HASH_NAME = 'hashes_train_partial.pkl'
TEST_HASH_NAME = 'hashes_test.pkl'
FILENAMES = dict(train=TRAIN_HASH_NAME, test=TEST_HASH_NAME)

def load_hashes(dataset):
    with open(FILENAMES[dataset]) as f:
        data = pickle.load(f)
    for k, v in data.iteritems():
        data[k] = np.around(v).astype('bool')
    return data

def get_full_image_name(partial):
    return '_'.join(partial.split('_')[:-2])

def get_image_label(x):
    return get_full_image_name(x).split('/')[-1]

def to_int(x):
    total = 0
    for i in xrange(len(x)):
        total *= 2
        total += x[i]
    return total

# get mapping from image names to partial names
def get_test_image_mapping(ds='test'):
    dir_file = os.path.join(OX_DIR, '.'.join([ds, 'txt']))
    with open(dir_file) as f:
        images = f.readlines()

    # strip trailing newlines
    test_images = set([i[:-1] for i in images])

    all_partials = os.listdir(IMAGE_DIR)

    mapping = collections.defaultdict(set)
    for partial in all_partials:
        if get_full_image_name(partial) in test_images:
            mapping[get_full_image_name(partial)].add(partial)

    return mapping


def do_lookups():
    lookup_table = collections.defaultdict(set)
    for fn, encoded in load_hashes('train').iteritems():
        lookup_table[to_int(encoded[0, :])].add(fn)

    image_potentials = {}
    test_table = load_hashes('test')
    tim = get_test_image_mapping()
    for image_name, partials in tim.iteritems():
        potentials = collections.Counter()
        for p in partials:
            hash_v = test_table[os.path.join(IMAGE_DIR, p)]
            hash_v = to_int(hash_v[0, :])

            # lookup the actual value
            candidates = map(get_image_label, lookup_table[hash_v])
            potentials.update(candidates)

            # lookup all 1 bit modifications
            for i in xrange(16):
                candidates = map(get_image_label, lookup_table[hash_v ^ (2 ** i)])
                potentials.update(candidates)

        image_potentials[image_name] = potentials
    return image_potentials

def calc_statistics(results):
    tim = get_test_image_mapping()
    mc = collections.Counter()
    for test_img, c in results.iteritems():
        mc.update([c.most_common(1)[0][0]])
    print mc

if __name__ == '__main__':
    results = do_lookups()
    calc_statistics(results)
