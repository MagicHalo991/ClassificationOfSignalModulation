"""Проверка файла датасета: формы, типы, уровни SNR, число сигналов на пару (класс, SNR).

Сохраняет results/classes_check.png — созвездия всех классов при +30 дБ
(новая constellation: один отсчёт на символ, сдвиг по максимуму мощности).
Запуск из корня проекта:  py -m scripts.check_dataset
"""

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from source import config  # noqa: E402
from source.dataset import load_labels, read_signals  # noqa: E402
from source.transforms import constellation  # noqa: E402

PLOT_SNR = 30


def main():
    path = config.RAW_FILE
    if not path.exists():
        raise SystemExit(f"нет файла {path}")

    print(f"Файл: {path} ({path.stat().st_size / 2**30:.2f} ГиБ)")
    with h5py.File(path, "r") as f:
        for key in ("X", "Y", "Z"):
            d = f[key]
            print(f"  {key}: shape={d.shape}, dtype={d.dtype}, chunks={d.chunks}, compression={d.compression}")

    labels, snrs = load_labels(path)
    levels = np.unique(snrs)
    print(f"\nУровни SNR ({len(levels)}): {levels.tolist()}")
    if levels.tolist() != list(config.SNRS):
        print("  ВНИМАНИЕ: уровни SNR не совпадают с config.SNRS")
    print(f"Классов: {len(np.unique(labels))} (ожидалось {config.NUM_CLASSES})")

    snr_idx = np.searchsorted(levels, snrs)
    counts = np.zeros((config.NUM_CLASSES, len(levels)), dtype=np.int64)
    np.add.at(counts, (labels, snr_idx), 1)
    print(f"\nСигналов на пару (класс, SNR): min={counts.min()}, max={counts.max()}")
    header = "класс \\ SNR".ljust(12) + "".join(f"{s:>6d}" for s in levels)
    print(header)
    for c, name in enumerate(config.CLASSES):
        print(name.ljust(12) + "".join(f"{v:>6d}" for v in counts[c]))

    # Первый сигнал каждого класса при PLOT_SNR
    picks = []
    for c in range(config.NUM_CLASSES):
        idx = np.flatnonzero((labels == c) & (snrs == PLOT_SNR))
        picks.append(idx[0])
    order = np.argsort(picks)
    with h5py.File(path, "r") as f:
        sig_sorted = read_signals(f, np.asarray(picks)[order])
    signals = np.empty_like(sig_sorted)
    signals[order] = sig_sorted

    fig, axes = plt.subplots(4, 6, figsize=(13, 9.5))
    for c, ax in enumerate(axes.flat):
        ax.imshow(constellation(signals[c]), cmap="magma",
                  extent=[-config.CONST_RANGE, config.CONST_RANGE] * 2)
        ax.set_title(f"{c}: {config.CLASSES[c]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Созвездия классов при SNR = +{PLOT_SNR} дБ: один отсчёт на символ (первый сигнал ячейки)")
    fig.tight_layout()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / "classes_check.png"
    fig.savefig(out, dpi=110)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
