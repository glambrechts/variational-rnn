import os
import gzip
import torch
import numpy as np

from urllib import request


SOURCE = "http://yann.lecun.com/exdb/mnist"
DESTINATION = "mnist"

TRAIN_FILENAME = "train-images-idx3-ubyte.gz"
TEST_FILENAME = "t10k-images-idx3-ubyte.gz"


def mnist(device='cpu'):

    os.makedirs(DESTINATION, exist_ok=True)

    # If archives are not saved, download them
    train_source = f"{SOURCE}/{TRAIN_FILENAME}"
    test_source = f"{SOURCE}/{TEST_FILENAME}"
    train_destination = f"{DESTINATION}/{TRAIN_FILENAME}"
    test_destination = f"{DESTINATION}/{TEST_FILENAME}"

    if not os.path.exists(train_destination) or not os.path.exists(test_destination):
        request.urlretrieve(train_source, train_destination)
        request.urlretrieve(test_source, test_destination)

    # Unzip images
    with gzip.open(train_destination, "rb") as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16)
        train_images = train_images.reshape(-1, 28,  28)
        train_images = (train_images / 255.0 - 0.5).astype(np.float32)
    with gzip.open(test_destination, "rb") as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16)
        test_images = test_images.reshape(-1, 28,  28)
        test_images = (test_images / 255.0 - 0.5).astype(np.float32)

    # Return datasets
    train_images = torch.from_numpy(train_images)
    test_images = torch.from_numpy(test_images)
    return train_images.to(device), test_images.to(device)
