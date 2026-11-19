"""Работа с датасетом RadioML 2018.01A: метки, чтение сигналов, разбиение, выборка для DataLoader.

Формат HDF5-файла:
    X — (N, 1024, 2) float32, отсчёты I/Q (около 20 ГБ, целиком в память не читается);
    Y — (N, 24) one-hot метки классов;
    Z — (N, 1) SNR в дБ.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from source import config
from source.transforms import to_complex

_LABEL_CHUNK = 1 << 16  # строк Y за одно чтение


# ---------------------------------------------------------------- чтение
def load_labels(path=None):
    """Возвращает (labels, snrs): номера классов (argmax по Y) и SNR каждого сигнала.

    Y читается порциями, чтобы не держать в памяти весь one-hot массив.
    """
    path = config.RAW_FILE if path is None else path
    with h5py.File(path, "r") as f:
        y, zds = f["Y"], f["Z"]
        n = y.shape[0]
        labels = np.empty(n, dtype=np.int16)
        for start in range(0, n, _LABEL_CHUNK):
            stop = min(start + _LABEL_CHUNK, n)
            labels[start:stop] = np.argmax(y[start:stop], axis=1)
        snrs = np.asarray(zds[:]).reshape(-1).astype(np.int16)
    return labels, snrs


def read_signals(h5file, indices):
    """Читает сигналы с заданными номерами и возвращает комплексный массив (len, N).

    h5file — открытый h5py.File; indices — строго возрастающие номера строк.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, h5file["X"].shape[1]), dtype=np.complex64)
    if indices.ndim != 1 or np.any(np.diff(indices) <= 0):
        raise ValueError("номера сигналов должны быть строго возрастающими")
    return to_complex(h5file["X"][indices])


# ---------------------------------------------------------------- разбиение
def make_split(labels, snrs, per_cell=None, fractions=None, seed=None):
    """Стратифицированное разбиение по парам (класс, SNR).

    В каждой паре случайно выбирается per_cell сигналов, которые делятся на
    train/val/test в пропорциях fractions (одинаковое число в каждой паре).

    Возвращает словарь:
        indices — номера выбранных сигналов в датасете (по возрастанию);
        labels, snrs — их классы и SNR;
        train, val, test — позиции внутри indices (по возрастанию).
    При одинаковом seed результат одинаков.
    """
    per_cell = config.SAMPLES_PER_CELL if per_cell is None else per_cell
    fractions = config.SPLIT_FRACTIONS if fractions is None else fractions
    seed = config.SEED if seed is None else seed

    labels = np.asarray(labels)
    snrs = np.asarray(snrs)
    if labels.shape != snrs.shape or labels.ndim != 1:
        raise ValueError("labels и snrs должны быть одномерными массивами одной длины")
    if abs(sum(fractions) - 1.0) > 1e-9 or len(fractions) != 3:
        raise ValueError("fractions — три доли, в сумме 1")

    n_train = int(round(fractions[0] * per_cell))
    n_val = int(round(fractions[1] * per_cell))
    if n_train + n_val > per_cell:
        raise ValueError("слишком маленький per_cell для заданных долей")

    # Группировка по ячейкам (класс, SNR) одной сортировкой
    snr_values, snr_idx = np.unique(snrs, return_inverse=True)
    key = labels.astype(np.int64) * len(snr_values) + snr_idx
    order = np.argsort(key, kind="stable")      # внутри ячейки — по возрастанию номера
    cell_keys, starts, counts = np.unique(key[order], return_index=True, return_counts=True)

    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for k, s, c in zip(cell_keys, starts, counts):
        if c < per_cell:
            cls, snr = divmod(int(k), len(snr_values))
            raise ValueError(f"в ячейке (класс {cls}, SNR {snr_values[snr]}) {c} сигналов < {per_cell}")
        chosen = order[s + rng.choice(c, size=per_cell, replace=False)]  # в случайном порядке
        parts["train"].append(chosen[:n_train])
        parts["val"].append(chosen[n_train:n_train + n_val])
        parts["test"].append(chosen[n_train + n_val:])

    parts = {name: np.concatenate(p) for name, p in parts.items()}
    indices = np.sort(np.concatenate(list(parts.values())))
    split = {
        "indices": indices,
        "labels": labels[indices],
        "snrs": snrs[indices],
    }
    for name, p in parts.items():
        split[name] = np.sort(np.searchsorted(indices, p))
    return split


def save_split(split, path=None):
    """Сохраняет разбиение в .npz."""
    path = config.SPLIT_FILE if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **split)


def load_split(path=None):
    """Загружает разбиение из .npz в словарь массивов."""
    path = config.SPLIT_FILE if path is None else path
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------- выборка для обучения
class ImageDataset(Dataset):
    """Map-style выборка изображений из кэша data/cache/<представление>.npy.

    part — "train", "val" или "test"; representations — имя или список имён представлений.
    Элемент: (изображение float32 (1, 64, 64) в диапазоне 0..1, метка, SNR);
    при нескольких представлениях вместо изображения — кортеж изображений.
    Файлы кэша открываются лениво (np.load с mmap_mode="r"), уже в процессе-воркере
    DataLoader, поэтому выборка безопасно работает при num_workers > 0.
    """

    def __init__(self, part, representations, split_file=None, cache_dir=None):
        if isinstance(representations, str):
            representations = (representations,)
        representations = tuple(representations)
        unknown = set(representations) - set(config.ALL_REPRESENTATIONS)
        if unknown:
            raise ValueError(f"неизвестные представления: {sorted(unknown)}")
        if part not in ("train", "val", "test"):
            raise ValueError(f"part должен быть train/val/test, получено {part!r}")

        split = load_split(config.SPLIT_FILE if split_file is None else split_file)
        self.part = part
        self.representations = representations
        self.cache_dir = config.CACHE_DIR if cache_dir is None else cache_dir
        self.n_total = len(split["indices"])
        self.positions = split[part].astype(np.int64)   # строки кэша
        self.labels = split["labels"][self.positions].astype(np.int64)
        self.snrs = split["snrs"][self.positions].astype(np.int64)
        self._arrays = None

        for rep in representations:
            if not (self.cache_dir / f"{rep}.npy").exists():
                raise FileNotFoundError(f"нет кэша {self.cache_dir / f'{rep}.npy'}; запустите scripts.build_cache")

    def _open(self):
        arrays = []
        for rep in self.representations:
            arr = np.load(self.cache_dir / f"{rep}.npy", mmap_mode="r")
            if arr.shape[0] != self.n_total:
                raise ValueError(f"в кэше {rep}.npy {arr.shape[0]} строк, а в разбиении {self.n_total}")
            arrays.append(arr)
        self._arrays = arrays

    def __getstate__(self):
        # memmap не передаём в воркеры: каждый откроет файлы сам
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        if self._arrays is None:
            self._open()
        row = self.positions[i]
        images = tuple(
            torch.from_numpy(np.asarray(arr[row], dtype=np.float32) / 255.0).unsqueeze(0)
            for arr in self._arrays
        )
        image = images[0] if len(images) == 1 else images
        return image, int(self.labels[i]), int(self.snrs[i])
