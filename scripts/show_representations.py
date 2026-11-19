"""Обзорные картинки всех четырёх представлений.

results/representations.png      — 6 классов при +30 дБ (строки) x 4 представления (столбцы);
results/representations_snr.png  — QPSK при SNR -10, 0, 10, 20, 30 дБ x 4 представления.

Сигналы берутся из разбиения (data/splits/split.npz): первый сигнал нужной пары (класс, SNR).
Запуск из корня проекта:  py -m scripts.show_representations
"""

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from source import config  # noqa: E402
from source.dataset import load_split, read_signals  # noqa: E402
from source.transforms import TRANSFORMS  # noqa: E402

CLASSES_SHOWN = ("BPSK", "QPSK", "16QAM", "64QAM", "FM", "AM-DSB-WC")
SNR_SHOWN = (-10, 0, 10, 20, 30)
SNR_CLASS = "QPSK"
REPS = config.ALL_REPRESENTATIONS


def pick(split, cls_name, snr):
    """Номер в датасете первого сигнала пары (класс, SNR) из разбиения."""
    c = config.CLASSES.index(cls_name)
    pos = np.flatnonzero((split["labels"] == c) & (split["snrs"] == snr))
    if pos.size == 0:
        raise ValueError(f"в разбиении нет сигналов {cls_name} при {snr} дБ")
    return int(split["indices"][pos[0]])


def read_many(h5, indices):
    """Читает сигналы в произвольном порядке номеров (read_signals требует возрастания)."""
    indices = np.asarray(indices)
    order = np.argsort(indices)
    out = np.empty((len(indices), config.SIGNAL_LEN), dtype=np.complex64)
    out[order] = read_signals(h5, indices[order])
    return out


def grid(signals, row_titles, title, path):
    fig, axes = plt.subplots(len(signals), len(REPS), figsize=(2.6 * len(REPS), 2.5 * len(signals)))
    for i, z in enumerate(signals):
        for j, rep in enumerate(REPS):
            ax = axes[i, j]
            ax.imshow(TRANSFORMS[rep](z), cmap="magma")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(rep, fontsize=11)
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=11)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Сохранено: {path}")


def main():
    if not config.SPLIT_FILE.exists():
        raise SystemExit(f"нет {config.SPLIT_FILE}; сначала запустите py -m scripts.make_split")
    split = load_split(config.SPLIT_FILE)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with h5py.File(config.RAW_FILE, "r") as h5:
        sig = read_many(h5, [pick(split, c, 30) for c in CLASSES_SHOWN])
        grid(sig, CLASSES_SHOWN, "Представления классов при SNR = +30 дБ",
             config.RESULTS_DIR / "representations.png")
        sig = read_many(h5, [pick(split, SNR_CLASS, s) for s in SNR_SHOWN])
        grid(sig, [f"{SNR_CLASS}, {s:+d} дБ" for s in SNR_SHOWN],
             f"{SNR_CLASS} при разных SNR", config.RESULTS_DIR / "representations_snr.png")


if __name__ == "__main__":
    main()
