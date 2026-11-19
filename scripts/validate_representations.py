"""Проверка представлений на настоящих данных при разных SNR.

Что считается (по кэшу data/cache и по исходному HDF5):
1. Различимость классов: доля верных ответов простейшего классификатора «ближайший
   центроид» (средние изображения классов по train, проверка на test) для каждого SNR.
   Это не модель проекта, а мера того, сколько информации о классе несёт само изображение.
2. Различимость сигнала и чистого шума: тот же классификатор на два класса
   (сигнал любого класса / комплексный белый шум). 50 % — неотличимо от шума.
3. Спектрограмма: оценка SNR по спектрограмме (QPSK) против SNR из разметки датасета.
4. Созвездие: EVM символьных отсчётов (BPSK, QPSK, 16QAM, 64QAM) против SNR.
5. Матрицы ошибок классификатора из п.1 при +30 дБ.

Результаты: results/validation/*.png и таблицы в консоли.
Запуск из корня проекта:  py -m scripts.validate_representations
"""

from multiprocessing import Pool

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from source import config  # noqa: E402
from source.dataset import load_split, read_signals  # noqa: E402
from source.transforms import TRANSFORMS, normalize_power, spectrogram_db, symbol_phase  # noqa: E402
from tests.test_transforms import evm_db  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "validation"
REPS = config.ALL_REPRESENTATIONS
SNRS = np.array(config.SNRS)
N_NOISE = 2000
# Категориальная палитра (проверенная, светлая тема), порядок фиксирован
COLORS = {"spectrogram": "#2a78d6", "scalogram": "#eb6834", "constellation": "#1baf7a", "iq_density": "#eda100"}
MARKERS = {"spectrogram": "o", "scalogram": "s", "constellation": "^", "iq_density": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


# ---------------------------------------------------------------- вычисления
def _noise_images(seed):
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal(config.SIGNAL_LEN) + 1j * rng.standard_normal(config.SIGNAL_LEN)).astype(np.complex64)
    return np.stack([TRANSFORMS[r](z) for r in REPS])


def nearest_centroid(train_x, train_y, test_x, n_classes):
    """Классификатор «ближайший центроид» (евклидово расстояние)."""
    cent = np.stack([train_x[train_y == c].mean(axis=0) for c in range(n_classes)])
    d = (test_x ** 2).sum(1)[:, None] - 2 * test_x @ cent.T + (cent ** 2).sum(1)[None, :]
    return np.argmin(d, axis=1)


def class_separability(split, caches):
    """acc[rep][snr] и матрицы ошибок при +30 дБ."""
    acc = {r: [] for r in REPS}
    conf = {}
    lab, snr = split["labels"].astype(int), split["snrs"]
    for s in SNRS:
        tr = split["train"][snr[split["train"]] == s]
        te = split["test"][snr[split["test"]] == s]
        for r in REPS:
            x = caches[r]
            pred = nearest_centroid(x[tr].reshape(len(tr), -1).astype(np.float32), lab[tr],
                                    x[te].reshape(len(te), -1).astype(np.float32), config.NUM_CLASSES)
            acc[r].append(float((pred == lab[te]).mean()))
            if s == 30:
                m = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES))
                np.add.at(m, (lab[te], pred), 1)
                conf[r] = m / m.sum(1, keepdims=True)
    return acc, conf


def noise_separability(split, caches, noise):
    """Сигнал против шума: balanced accuracy для каждого SNR."""
    rng = np.random.default_rng(0)
    half = 900                      # в test на один SNR 24 x 39 = 936 сигналов
    acc = {r: [] for r in REPS}
    snr = split["snrs"]
    for s in SNRS:
        tr = rng.choice(split["train"][snr[split["train"]] == s], half, replace=False)
        te = rng.choice(split["test"][snr[split["test"]] == s], half, replace=False)
        for i, r in enumerate(REPS):
            x = caches[r]
            n = noise[:, i].reshape(len(noise), -1).astype(np.float32)
            trx = np.concatenate([x[tr].reshape(half, -1).astype(np.float32), n[:half]])
            tex = np.concatenate([x[te].reshape(half, -1).astype(np.float32), n[half:2 * half]])
            y = np.r_[np.ones(half, int), np.zeros(half, int)]
            pred = nearest_centroid(trx, y, tex, 2)
            acc[r].append(float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())))
    return acc


def first_rows(split, cls, s, n):
    pos = np.flatnonzero((split["labels"] == config.CLASSES.index(cls)) & (split["snrs"] == s))[:n]
    return split["indices"][pos]


def spectrogram_snr(split, h5, n=48):
    """Оценка SNR по спектрограмме QPSK: мощность над шумовым полом / шум (вне полосы |f| > 0.25)."""
    est = []
    for s in SNRS:
        vals = []
        for z in read_signals(h5, first_rows(split, "QPSK", s, n)):
            p = (10 ** (spectrogram_db(z) / 10)).mean(axis=1)
            floor = np.r_[p[:16], p[-16:]].mean()
            excess = p.sum() - floor * 64
            # сигнал не выделяется над шумовым полом -> оценки нет (NaN)
            vals.append(10 * np.log10(excess / (floor * 64)) if excess > 0 else np.nan)
        vals = np.array(vals)
        est.append(float(np.nanmedian(vals)) if np.mean(np.isnan(vals)) < 0.5 else float("nan"))
    return est


def constellation_evm(split, h5, mods=("BPSK", "QPSK", "16QAM", "64QAM"), n=32):
    out = {}
    for mod in mods:
        vals = []
        for s in SNRS:
            e = []
            for z in read_signals(h5, first_rows(split, mod, s, n)):
                zn = normalize_power(z)
                e.append(evm_db(zn[symbol_phase(zn)::config.SAMPLES_PER_SYMBOL], mod))
            vals.append(float(np.median(e)))
        out[mod] = vals
    return out


# ---------------------------------------------------------------- графики
def _style(ax, xlabel, ylabel):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)


def _label_positions(values, min_gap):
    """Разводит подписи концов линий по вертикали, чтобы они не слипались."""
    order = np.argsort(values)
    pos = np.array(values, float)
    for a, b in zip(order[:-1], order[1:]):
        if pos[b] - pos[a] < min_gap:
            pos[b] = pos[a] + min_gap
    return pos


def plot_lines(series, title, ylabel, path, ref=None, ref_label=None, ylim=None, percent=True,
               refs=(), legend_loc="upper left"):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ends = []
    for name, ys in series.items():
        ys = np.array(ys, float) * (100 if percent else 1)
        ax.plot(SNRS, ys, color=COLORS.get(name, INK), marker=MARKERS.get(name, "o"), markersize=6,
                linewidth=2, label=name)
        ends.append((name, ys[~np.isnan(ys)][-1]))
    if ref is not None:
        refs = ((ref, ref_label, "--"),) + tuple(refs)
    for y, lab, ls in refs:
        ax.plot(SNRS, y, color=MUTED, linestyle=ls, linewidth=1.2, label=lab)
    if ylim:
        ax.set_ylim(*ylim)
    lo, hi = ax.get_ylim()
    ypos = _label_positions([v for _, v in ends], 0.045 * (hi - lo))
    for (name, v), y in zip(ends, ypos):
        ax.annotate(name, (SNRS[-1], v), xytext=(SNRS[-1] + 1.2, y), textcoords="data",
                    va="center", fontsize=9, color=INK)
    _style(ax, "SNR из разметки датасета, дБ", ylabel)
    ax.set_xlim(SNRS[0] - 1, SNRS[-1] + 10)
    ax.set_title(title, color=INK, loc="left")
    ax.legend(frameon=False, loc=legend_loc, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(conf, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    for ax, r in zip(axes.flat, REPS):
        ax.imshow(conf[r] * 100, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(config.NUM_CLASSES), config.CLASSES, rotation=90, fontsize=7)
        ax.set_yticks(range(config.NUM_CLASSES), config.CLASSES, fontsize=7)
        ax.set_title(f"{r}: верно {np.trace(conf[r]) / config.NUM_CLASSES * 100:.0f} %", color=INK)
        ax.set_xlabel("предсказанный класс", color=MUTED)
        ax.set_ylabel("истинный класс", color=MUTED)
    fig.suptitle("Матрицы ошибок «ближайшего центроида» при +30 дБ, % по строке", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- запуск
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = load_split()
    caches = {r: np.load(config.CACHE_DIR / f"{r}.npy", mmap_mode="r") for r in REPS}

    print("Шумовые изображения ...", flush=True)
    with Pool() as pool:
        noise = np.stack(pool.map(_noise_images, range(N_NOISE), chunksize=50))

    print("Различимость классов ...", flush=True)
    cls_acc, conf = class_separability(split, caches)
    print("Различимость сигнала и шума ...", flush=True)
    noise_acc = noise_separability(split, caches, noise)
    with h5py.File(config.RAW_FILE, "r") as h5:
        print("SNR по спектрограмме ...", flush=True)
        spec_snr = spectrogram_snr(split, h5)
        print("EVM созвездия ...", flush=True)
        evm = constellation_evm(split, h5)

    # --- таблицы
    hdr = "SNR    " + "".join(f"{r:>15s}" for r in REPS)
    print("\n1. Ближайший центроид, доля верных ответов (24 класса, случайно 4.2 %):")
    print(hdr)
    for i, s in enumerate(SNRS):
        print(f"{s:+4d}   " + "".join(f"{cls_acc[r][i] * 100:14.1f}%" for r in REPS))
    print("\n2. Сигнал против шума, balanced accuracy (50 % = неотличимо):")
    print(hdr)
    for i, s in enumerate(SNRS):
        print(f"{s:+4d}   " + "".join(f"{noise_acc[r][i] * 100:14.1f}%" for r in REPS))
    print("\n3. SNR по спектрограмме (QPSK) и 4. EVM созвездия, дБ:")
    print("SNR   оценка_спектр  " + "".join(f"{m:>10s}" for m in evm))
    for i, s in enumerate(SNRS):
        print(f"{s:+4d}   {spec_snr[i]:+12.1f}  " + "".join(f"{evm[m][i]:+10.1f}" for m in evm))
    print("\n5. Верно при +30 дБ по классам:")
    print("класс       " + "".join(f"{r:>15s}" for r in REPS))
    for c, name in enumerate(config.CLASSES):
        print(f"{name:11s} " + "".join(f"{conf[r][c, c] * 100:14.0f}%" for r in REPS))

    # --- графики
    plot_lines(cls_acc, "Различимость 24 классов: ближайший центроид", "верных ответов, %",
               OUT_DIR / "class_separability.png", ref=np.full(len(SNRS), 100 / 24),
               ref_label="случайное угадывание", ylim=(0, 100))
    plot_lines(noise_acc, "Сигнал против белого шума", "balanced accuracy, %",
               OUT_DIR / "noise_separability.png", ref=np.full(len(SNRS), 50.0),
               ref_label="неотличимо от шума", ylim=(40, 102))
    plot_lines({"QPSK": spec_snr}, "SNR, измеренный по спектрограмме, против разметки датасета",
               "измеренный SNR, дБ", OUT_DIR / "spectrogram_snr.png", ref=SNRS.astype(float),
               ref_label="y = x (совпадение с разметкой)", percent=False,
               refs=((2.0 * SNRS, "y = 2x", ":"),), ylim=(-45, 65))
    evm_series = {m: v for m, v in evm.items()}
    fig_colors = dict(zip(evm_series, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    COLORS.update(fig_colors)
    plot_lines(evm_series, "EVM символьных отсчётов (constellation)", "EVM, дБ",
               OUT_DIR / "constellation_evm.png", ref=-SNRS.astype(float), ref_label="EVM = −SNR (без искажений)",
               percent=False, ylim=(-35, 5), legend_loc="lower left")
    plot_confusion(conf, OUT_DIR / "confusion_30dB.png")
    print(f"\nГрафики: {OUT_DIR}")


if __name__ == "__main__":
    main()
