import itertools
import os
import random

import numpy as np
from scipy.misc import imread, imresize

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

TRAIN_IMAGE_NAMES = load_patch_names('train')
TEST_IMAGE_NAMES = load_patch_names('test')

f = lambda name: imresize(
    imread(
        os.path.join(IMAGE_DIR, name),
        flatten=True
    ).astype('uint8'),
    0.5
)

#TRAIN_IMAGES = map(f, TRAIN_IMAGE_NAMES)
#print 'done loading train images'

TEST_IMAGES = map(f, TEST_IMAGE_NAMES)
print 'done loading test images'


def image_iterator(_img, patch_size=16, bw=True, max_patches=BATCH_SIZE):
    img = _img.copy()
    img /= 255.0

    end1 = img.shape[0] - patch_size
    end2 = img.shape[1] - patch_size

    all_patches = []
    for start1 in xrange(end1):
        for start2 in xrange(end2):
            all_patches.append(img[start1:(start1 + patch_size),
                                   start2:(start2 + patch_size)])

    for patch in random.sample(all_patches, min(len(all_patches), max_patches)):
        yield patch


def img_list_iterator(L, idxs, *args, **kwargs):
    return itertools.chain(
        *map(
            lambda x: image_iterator(L[x], *args, **kwargs),
            idxs
        )
    )

def infinite_patch_iterator(L, *args, **kwargs):
    idxs = range(len(L))
    while True:
        random.shuffle(idxs)
        for x in img_list_iterator(L, idxs, *args, **kwargs):
            yield x


class OxfordImages(Dataset):
    def __init__(self, which_set):
        assert which_set in ['train', 'test']
        self.which_set = which_set

    def _iterator(self, batch_size=100, num_batches=100, **kwargs):
        if self.which_set == 'test':
            ipi = infinite_patch_iterator(TEST_IMAGES)
        else:
            ipi = infinite_patch_iterator(TRAIN_IMAGES)

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
    ds = OxfordImages('test')
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

    trainer = Train(ds, ae, algorithm=alg, save_path='model2.pkl', save_freq=1)
    trainer.main_loop()


if __name__ == '__main__':
    train()
