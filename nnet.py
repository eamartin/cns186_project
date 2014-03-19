import itertools
import os
import cPickle as pickle
import random

import numpy as np
from scipy.misc import imread
import theano
import theano.tensor as T

from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.corruption import GaussianCorruptor
from pylearn2.datasets.dataset import Dataset
from pylearn2.models.autoencoder import DenoisingAutoencoder
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

OX_DIR = '/home/emartin/186_data/oxford/'
IMAGE_DIR = '/home/emartin/186_data/oxford/partials/'

PATCH_SIZE = 16 ** 2
REP_SIZE = 16

GAUSS_NOISE = 0.5

LEARNING_RATE = 0.25
MOMENTUM = 0.5
MAX_EPOCHS = 200
BATCHES_PER_EPOCH = 5000
BATCH_SIZE = 150

def image_iterator(filename, patch_size=16, bw=True):
    img = imread(filename, flatten=bw)
    img /= 255.0

    end1 = img.shape[0] - patch_size
    end2 = img.shape[1] - patch_size

    for start1 in xrange(end1):
        for start2 in xrange(end2):
            yield img[start1:(start1 + patch_size),
                      start2:(start2 + patch_size)]

def img_list_iterator(L, *args, **kwargs):
    return itertools.chain(
        *map(
            lambda x: image_iterator(x, *args, **kwargs),
            L
        )
    )

def infinite_patch_iterator(L, *args, **kwargs):
    while True:
        my_list = L[:]
        random.shuffle(my_list)
        for x in img_list_iterator(my_list, *args, **kwargs):
            yield x


class OxfordImages(Dataset):
    def __init__(self, which_set):
        assert which_set in ['train', 'test']
        self.which_set = which_set
        self.patch_names = self.load_patch_names(self.which_set)

    @staticmethod
    def load_patch_names(which_set):
        dir_file = os.path.join(OX_DIR, '.'.join([which_set, 'txt']))
        with open(dir_file) as f:
            images = f.readlines()

        # strip trailing newlines
        images = set([i[:-1] for i in images])

        all_segments = os.listdir(IMAGE_DIR)
        set_segments = filter(lambda x: '_'.join(x.split('_')[:-2]) in images,
                              all_segments)
        return set_segments

    def _iterator(self, batch_size=100, num_batches=100, **kwargs):
        patch_files = map(
            lambda x: os.path.join(IMAGE_DIR, x),
            self.patch_names
        )

        ipi = infinite_patch_iterator(patch_files)
        for batchno in xrange(num_batches):
            batch = []
            for _ in xrange(batch_size):
                batch.append(ipi.next())
            yield (np.array(map(lambda x: x.reshape(x.size), batch)),)

    def iterator(self, batch_size=100, num_batches=100, **kwargs):
        it = self._iterator(batch_size=batch_size,
                            num_batches=num_batches,
                            **kwargs)

        class ItWrapper(object):
            stochastic = False

            def __init__(self, it):
                self.it = it
                self.batch_size = batch_size
                self.remaining_batches = num_batches
                self.num_examples = batch_size * num_batches

            def next(self):
                self.remaining_batches -= 1
                return self.it.next()

            def __iter__(self):
                return self

        return ItWrapper(it)


def train():
    ds = OxfordImages('train')
    ae = DenoisingAutoencoder(GaussianCorruptor(GAUSS_NOISE),
                              PATCH_SIZE, REP_SIZE, 'tanh', 'tanh')

    cost = MeanSquaredReconstructionError()

    alg = SGD(
        LEARNING_RATE,
        init_momentum=MOMENTUM,
        cost=cost,
        termination_criterion=EpochCounter(MAX_EPOCHS),
        batches_per_iter=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        monitoring_dataset=ds,
        monitoring_batches=10
    )

    trainer = Train(ds, ae, algorithm=alg, save_path='model.pkl', save_freq=1)
    trainer.main_loop()

def get_encoder():
    with open('model.pkl') as f:
        ae = pickle.load(f)

    in_patch = T.matrix()
    out = ae.encode(in_patch)
    encoder = theano.function([in_patch], out)
    return encoder

def evaluate(ds_name='test'):
    ds = OxfordImages(ds_name)
    filenames = map(lambda x: os.path.join(IMAGE_DIR, x), ds.patch_names)

    encoder = get_encoder()

    D = {}

    for idx, fn in enumerate(filenames):
        accum = np.zeros((1, 16))
        patch_count = 0

        batch = []
        for patch in image_iterator(fn):
            patch_count += 1
            batch.append(patch.reshape(patch.size))

            if len(batch) % 100 == 0:
                accum += np.around((encoder(np.array(batch)) + 1.0)/ 2.0).sum(axis=0)
                batch = []

        if batch != []:
            accum += np.around((encoder(np.array(batch)) + 1.0)/ 2.0).sum(axis=0)

        print 'done with image %s, %d out of %d' % (fn, idx, len(filenames))
        D[fn] = accum / patch_count

        if idx % 100 == 0:
            with open('hashes_%s_tmp.pkl' % ds_name, 'w') as f:
                pickle.dump(D, f)
            print 'TEMP SAVE'

    with open('hashes_%s.pkl' % ds_name, 'w') as f:
        pickle.dump(D, f)

if __name__ == '__main__':
    evaluate('train')
