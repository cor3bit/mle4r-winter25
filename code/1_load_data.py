import os

import datasets
import numpy as np

import sklearn.datasets as skdata
import libsvmdata
import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(dataset_id):
    if dataset_id == 'my_dataset':
        x = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return x, y
    else:
        raise ValueError(f'Unknown dataset_id: {dataset_id}')


if __name__ == '__main__':
    np.random.seed(1337)

    # ---------------- Data Loading ----------------
    # Simple signature
    dataset_id = 'my_dataset'
    x, y = load_data(dataset_id)

    # Sklearn datasets
    iris = skdata.load_iris()  # small tabular dataset
    x, y = iris.data, iris.target

    news = skdata.fetch_20newsgroups()
    x, y = news.data, news.target  # text dataset

    # LibSVM
    # Beware of the format of the data! Here, sparse data is returned
    x, y = libsvmdata.fetch_libsvm('a9a', normalize=True, verbose=True)

    # TensorFlow datasets
    tf.config.set_visible_devices([], device_type='GPU')  # "conceal" GPUs from TFDS
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'data')
    data, info = tfds.load(
        name='mnist',
        split=['train', ],
        batch_size=-1,
        data_dir=data_folder,
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )  # Beware of the format of the data! Here, tensors are returned
    data = tfds.as_numpy(data)
    x, y = data[0]

    # HuggingFace datasets
    data = datasets.load_dataset(
        'lizziepikachu/starwars_planets',
        cache_dir=data_folder,
    )
    df = data['train'].to_pandas()

    y = df['population'].to_numpy()
    x = df.drop(columns=['population']).to_numpy()


    # ---------------- Data Processing ----------------
    # TODO

    # ---------------- Batching ----------------
    class SimpleDataLoader:
        def __init__(self, x, y, batch_size):
            self.x = x
            self.y = y
            self.batch_size = batch_size
            self.n = len(x)
            self.indices = np.arange(self.n)

        def __iter__(self):
            for i in range(0, self.n, self.batch_size):
                batch_indices = self.indices[i:i + self.batch_size]
                yield self.x[batch_indices], self.y[batch_indices]


    batch_size = 32
    dataloader = SimpleDataLoader(x, y, batch_size)
    for x_batch, y_batch in dataloader:
        # do something with the batch
        pass

    # TODO poorman's data loader from disk

    # But! We need shuffling, infinite iterations (epochs) and reading from disk
