"""Построение стратифицированного разбиения train/val/test и сохранение в SPLIT_FILE.

Запуск из корня проекта:  py -m scripts.make_split
"""

import numpy as np

from source import config
from source.dataset import load_labels, make_split, save_split


def main():
    print(f"Чтение меток из {config.RAW_FILE} ...")
    labels, snrs = load_labels(config.RAW_FILE)
    split = make_split(labels, snrs, config.SAMPLES_PER_CELL, config.SPLIT_FRACTIONS, config.SEED)
    save_split(split, config.SPLIT_FILE)

    n_cells = len(np.unique(split["labels"])) * len(np.unique(split["snrs"]))
    print(f"Сохранено: {config.SPLIT_FILE}")
    print(f"Ячеек (класс, SNR): {n_cells}, сигналов на ячейку: {config.SAMPLES_PER_CELL}")
    print(f"Всего выбрано: {len(split['indices'])}")
    for part in ("train", "val", "test"):
        n = len(split[part])
        print(f"  {part:5s}: {n:7d}  ({n // n_cells} на ячейку)")


if __name__ == "__main__":
    main()
