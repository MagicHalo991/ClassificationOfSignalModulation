"""Иллюстрации трёх представлений для документации (папка docs/images).

docs/images/overview_30dB.png         — основные модуляции x 3 представления при +30 дБ;
docs/images/<модуляция>.png           — 3 представления x SNR (-10, 0, 10, 20, 30 дБ), с осями;
docs/images/raw/<модуляция>_<вид>.png — само изображение 64x64 при +30 дБ (увеличено в 4 раза),
                                        как его видит нейросеть.
Сигналы — первые сигналы соответствующих пар (класс, SNR) из разбиения.
Запуск из корня проекта:  py -m scripts.save_docs_images
"""

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pywt  # noqa: E402

from scripts.show_representations import pick, read_many  # noqa: E402
from source import config  # noqa: E402
from source.dataset import load_split  # noqa: E402
from source.transforms import TRANSFORMS  # noqa: E402

DOCS_DIR = config.ROOT / "docs" / "images"
MODS = ("OOK", "4ASK", "BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "AM-DSB-WC", "FM", "GMSK")
SNR_LIST = (-10, 0, 10, 20, 30)
REPS = config.REPRESENTATIONS
TITLES = {"spectrogram": "Спектрограмма", "scalogram": "Скалограмма", "constellation": "Созвездие"}
CMAP = "magma"
UPSCALE = 4


def _file_name(mod):
    return mod.replace("-", "_")


def draw(ax, rep, img, axes_labels=True):
    """Рисует изображение с осями в физических единицах."""
    n = config.SIGNAL_LEN
    if rep == "spectrogram":
        ax.imshow(img, cmap=CMAP, aspect="auto", extent=[0, n, -0.5, 0.5])
        if axes_labels:
            ax.set_xlabel("время, отсчёты")
            ax.set_ylabel("частота, доли fs")
    elif rep == "scalogram":
        ax.imshow(img, cmap=CMAP, aspect="auto", extent=[0, n, config.IMG_SIZE, 0])
        if axes_labels:
            scales = np.geomspace(config.CWT_SCALE_MIN, config.CWT_SCALE_MAX, config.IMG_SIZE)
            freqs = pywt.scale2frequency(config.CWT_WAVELET, scales)
            rows = [0, 21, 42, 63]
            ax.set_yticks([r + 0.5 for r in rows], [f"{freqs[r]:.3f}" for r in rows])
            ax.set_xlabel("время, отсчёты")
            ax.set_ylabel("частота, доли fs")
    else:
        r = config.CONST_RANGE
        ax.imshow(img, cmap=CMAP, extent=[-r, r, -r, r])
        if axes_labels:
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
    if not axes_labels:
        ax.set_xticks([])
        ax.set_yticks([])


def main():
    split = load_split()
    (DOCS_DIR / "raw").mkdir(parents=True, exist_ok=True)
    with h5py.File(config.RAW_FILE, "r") as h5:
        # --- сводка при +30 дБ
        sig30 = read_many(h5, [pick(split, m, 30) for m in MODS])
        fig, axes = plt.subplots(len(MODS), len(REPS), figsize=(9, 2.3 * len(MODS)))
        for i, (mod, z) in enumerate(zip(MODS, sig30)):
            for j, rep in enumerate(REPS):
                img = TRANSFORMS[rep](z)
                draw(axes[i, j], rep, img, axes_labels=False)
                if i == 0:
                    axes[i, j].set_title(TITLES[rep])
                if j == 0:
                    axes[i, j].set_ylabel(mod, fontsize=12)
                big = np.kron(img, np.ones((UPSCALE, UPSCALE), np.uint8))
                plt.imsave(DOCS_DIR / "raw" / f"{_file_name(mod)}_{rep}.png", big, cmap=CMAP, vmin=0, vmax=255)
        fig.suptitle("Основные модуляции при SNR = +30 дБ", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(DOCS_DIR / "overview_30dB.png", dpi=100)
        plt.close(fig)
        print(f"Сохранено: {DOCS_DIR / 'overview_30dB.png'}")

        # --- по каждой модуляции: представления x SNR
        for mod in MODS:
            sig = read_many(h5, [pick(split, mod, s) for s in SNR_LIST])
            fig, axes = plt.subplots(len(REPS), len(SNR_LIST), figsize=(3.2 * len(SNR_LIST), 3.0 * len(REPS)))
            for j, (s, z) in enumerate(zip(SNR_LIST, sig)):
                for i, rep in enumerate(REPS):
                    ax = axes[i, j]
                    draw(ax, rep, TRANSFORMS[rep](z), axes_labels=True)
                    if j > 0:
                        ax.set_ylabel("")
                    if i == 0:
                        ax.set_title(f"SNR {s:+d} дБ")
                    if j == 0:
                        ax.set_ylabel(f"{TITLES[rep]}\n" + ax.get_ylabel(), fontsize=10)
            fig.suptitle(f"{mod}: спектрограмма, скалограмма и созвездие при разных SNR")
            fig.tight_layout()
            out = DOCS_DIR / f"{_file_name(mod)}.png"
            fig.savefig(out, dpi=90)
            plt.close(fig)
            print(f"Сохранено: {out}")
    print(f"Изображения 64x64: {DOCS_DIR / 'raw'} ({len(MODS) * len(REPS)} файлов)")


if __name__ == "__main__":
    main()
