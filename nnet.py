import itertools
import os
import random

import numpy as np
from scipy.misc import imread

from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.corruption import GaussianCorruptor
from pylearn2.datasets.dataset import Dataset
from pylearn2.models.autoencoder import DenoisingAutoencoder
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

def image_iterator(filename, patch_size=16, bw=True):
    img = imread(filename, flatten=bw)

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


OX_DIR = '/home/emartin/186_data/oxford/'
IMAGE_DIR = '/home/emartin/186_data/oxford/partials/'
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

    def iterator(self, batch_size=100, num_batches=100):
        patch_files = map(
            lambda x: os.path.join(IMAGE_DIR, x),
            self.patch_names
        )

        ipi = infinite_patch_iterator(patch_files)
        for batchno in xrange(num_batches):
            batch = []
            for _ in xrange(batch_size):
                batch.append(ipi.next())
            yield np.array(map(lambda x: x.reshape(x.size), batch))

if __name__ == '__main__':
    oi = OxfordImages('train')
    i = oi.iterator(num_batches=2, batch_size=5)
    print i.next().shape


PATCH_SIZE = 16 ** 2
REP_SIZE = 16

GAUSS_NOISE = 0.3

LEARNING_RATE = 0.25 / PATCH_SIZE
MOMENTUM = 0.5
MAX_EPOCHS = 1
BATCHES_PER_EPOCH = 100
BATCH_SIZE = 64

def train():
    ae = DenoisingAutoencoder(GaussianCorruptor(GAUSS_NOISE),
                              PATCH_SIZE, REP_SIZE, 'tanh', 'tanh')

    cost = MeanBinaryCrossEntropy()

    alg = SGD(
        LEARNING_RATE,
        init_momentum=MOMENTUM,
        cost=c,
        termination_criterion=EpochCounter(MAX_EPOCHS),
        batches_per_iter=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        monitoring_dataset=ds,
        monitoring_batches=10
   )

    Train(ds, ae, algorithm=alg, save_path='model.pkl', save_freq=1)
