import numpy as np

from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.corruption import GaussianCorruptor
from pylearn2.datasets.dataset import Dataset
from pylearn2.models.autoencoder import DenoisingAutoencoder
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

class OxfordImages(Dataset):
    IMAGE_DIR = '/home/emartin/186_data/oxford/partials/'
    def __init__(self, which_set):
        assert which_set in ['train', 'test']
        self.load_images()

    def load_images():
        pass

    def iterator(self, batch_size=100, num_batches=100):
        return


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
