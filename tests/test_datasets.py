import gzip
import hashlib
import io
import pickle
import tarfile

import numpy as np
import pytest

from dpbench.core.registry import DATASETS
from dpbench.datasets import adult, cifar10, mnist
from dpbench.datasets.base import md5sum, subsample


def test_synthetic_dataset(tiny_dataset):
    assert tiny_dataset.X_train.dtype == np.float32
    assert tiny_dataset.input_shape == (8,)
    assert set(np.unique(tiny_dataset.groups_train)) == {0, 1}
    summary = tiny_dataset.summary()
    assert summary["n_train"] + summary["n_test"] == 600


def test_digits_dataset_is_bundled():
    dataset = DATASETS.get("digits")().load()
    assert dataset.input_shape == (1, 8, 8)
    assert dataset.n_train == 1437 and dataset.n_test == 360
    assert 0.0 <= dataset.X_train.min() and dataset.X_train.max() <= 1.0


def test_subsample(tiny_dataset):
    small = subsample(tiny_dataset, 50, 20, seed=0)
    assert small.n_train == 50 and small.n_test == 20 and len(small.groups_train) == 50
    assert subsample(tiny_dataset, 50, 20, seed=0).X_train.tobytes() == small.X_train.tobytes()


def write_idx(path, array):
    header = bytes([0, 0, 0x08, array.ndim]) + b"".join(int(s).to_bytes(4, "big") for s in array.shape)
    with gzip.open(path, "wb") as handle:
        handle.write(header + array.astype(np.uint8).tobytes())


def test_idx_reader_and_mnist_loader(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    arrays = {
        "train-images-idx3-ubyte.gz": rng.integers(0, 256, (30, 28, 28)),
        "train-labels-idx1-ubyte.gz": rng.integers(0, 10, 30),
        "t10k-images-idx3-ubyte.gz": rng.integers(0, 256, (10, 28, 28)),
        "t10k-labels-idx1-ubyte.gz": rng.integers(0, 10, 10),
    }
    for name, array in arrays.items():
        write_idx(tmp_path / name, array)
    np.testing.assert_array_equal(
        mnist.read_idx(tmp_path / "train-images-idx3-ubyte.gz"), arrays["train-images-idx3-ubyte.gz"]
    )
    monkeypatch.setattr(mnist, "download", lambda filename, mirrors, directory, md5=None: tmp_path / filename)
    dataset = DATASETS.get("mnist")(normalize=False).load()
    assert dataset.X_train.shape == (30, 1, 28, 28) and dataset.n_test == 10
    np.testing.assert_allclose(dataset.X_train[0, 0], arrays["train-images-idx3-ubyte.gz"][0] / 255.0, rtol=1e-6)


def test_idx_reader_rejects_bad_files(tmp_path):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(bytes([0, 0, 0x08, 1]) + (5).to_bytes(4, "big") + b"abc")
    with pytest.raises(ValueError):
        mnist.read_idx(path)


ADULT_ROWS = """39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Female, 0, 0, 40, United-States, >50K
53, ?, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K
28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, >50K
37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, >50K
49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K
52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K
31, Private, 45781, Masters, 14, Never-married, Prof-specialty, Not-in-family, White, Female, 14084, 0, 50, United-States, >50K
42, Private, 159449, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 5178, 0, 40, United-States, <=50K
"""


def test_adult_loader_on_a_sample(tmp_path, monkeypatch):
    path = tmp_path / "adult-all.csv"
    path.write_text(ADULT_ROWS)
    monkeypatch.setattr(adult, "download", lambda filename, mirrors, directory, md5=None: path)
    dataset = DATASETS.get("adult")(test_size=0.33).load()
    assert dataset.n_train + dataset.n_test == 9  # the record with a missing value is dropped
    assert dataset.input_shape == (105,)
    assert dataset.X_train.min() >= 0.0 and dataset.X_train.max() <= 1.0
    assert dataset.group_names == ["Female", "Male"]
    features, names = adult.encode_features(adult._read(path).dropna())
    assert features.shape[1] == len(names) == 105


def fake_cifar(tmp_path, batches, protocol=2):
    archive = tmp_path / f"cifar-{protocol}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in batches.items():
            data = payload if isinstance(payload, bytes) else pickle.dumps(payload, protocol=protocol)
            info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


@pytest.mark.parametrize("protocol", [2, 4])
def test_cifar_batches(tmp_path, protocol):
    rng = np.random.default_rng(0)
    batch = {"data": rng.integers(0, 256, (4, 3072)).astype(np.uint8), "labels": [1, 2, 3, 4]}
    archive = fake_cifar(tmp_path, {"data_batch_1": batch}, protocol)
    images, labels = cifar10.read_batches(archive, ["data_batch_1"])
    assert images.shape == (4, 3, 32, 32)
    assert labels.tolist() == [1, 2, 3, 4]


class _Exploit:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_cifar_reader_refuses_code_in_pickles(tmp_path):
    archive = fake_cifar(tmp_path, {"data_batch_1": pickle.dumps({"data": _Exploit(), "labels": []}, protocol=2)})
    with pytest.raises(pickle.UnpicklingError):
        cifar10.read_batches(archive, ["data_batch_1"])


def test_md5(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"dp-bench")
    assert md5sum(path) == hashlib.md5(b"dp-bench").hexdigest()


@pytest.mark.network
@pytest.mark.parametrize(
    "name, n_train, shape",
    [
        ("adult", 36177, (105,)),
        ("mnist", 60000, (1, 28, 28)),
        ("fashion_mnist", 60000, (1, 28, 28)),
    ],
)
def test_real_datasets(name, n_train, shape):
    dataset = DATASETS.get(name)().load()
    assert dataset.n_train == n_train
    assert dataset.input_shape == shape
