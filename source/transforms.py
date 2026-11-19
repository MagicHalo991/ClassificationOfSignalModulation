"""Преобразования комплексного радиосигнала в изображения 64x64 (uint8).

Каждая функция-представление принимает один комплексный сигнал длиной SIGNAL_LEN,
нормирует его мощность к единице и возвращает массив uint8 формы (IMG_SIZE, IMG_SIZE),
растянутый по яркости на диапазон 0..255. Постоянное изображение превращается в нули.

Ориентация всех изображений: строка 0 — верх картинки.
"""

import numpy as np
import pywt

from source import config

_EPS = 1e-12


# ---------------------------------------------------------------- вспомогательные
def to_complex(iq):
    """Массив (..., N, 2) с каналами I/Q -> комплексный массив (..., N) complex64."""
    iq = np.asarray(iq)
    if iq.shape[-1] != 2:
        raise ValueError(f"последняя ось должна иметь длину 2 (I, Q), получено {iq.shape}")
    return (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64)


def normalize_power(z):
    """Нормирует среднюю мощность сигнала к единице (нулевой сигнал не меняется)."""
    z = np.asarray(z, dtype=np.complex128)
    p = np.mean(np.abs(z) ** 2)
    return z / np.sqrt(p) if p > 0 else z


def to_uint8(img):
    """Min-max растяжение на 0..255; постоянное изображение -> нули."""
    img = np.asarray(img, dtype=np.float64)
    lo, hi = img.min(), img.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 0:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.round((img - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _check_signal(z):
    z = np.asarray(z)
    if z.ndim != 1 or not np.iscomplexobj(z):
        raise ValueError(f"ожидается один комплексный сигнал формы (N,), получено {z.shape}, {z.dtype}")
    return z


# ---------------------------------------------------------------- спектрограмма
_HANN = np.hanning(config.STFT_WINDOW + 1)[:-1]  # периодическое окно Ханна


def spectrogram_db(z):
    """Спектрограмма в дБ (float, до перевода в uint8): STFT, окно Ханна 64, шаг 16.

    Сигнал дополняется отражением по краям так, чтобы получилось ровно IMG_SIZE окон.
    Строки — частота (положительные частоты сверху), столбцы — время.
    """
    z = normalize_power(_check_signal(z))
    win, hop, n_frames = config.STFT_WINDOW, config.STFT_HOP, config.IMG_SIZE
    total = (n_frames - 1) * hop + win          # нужная длина после дополнения
    pad = total - len(z)
    if pad < 0:
        raise ValueError(f"сигнал длиннее {total} отсчётов не помещается в {n_frames} окон")
    zp = np.pad(z, (pad // 2, pad - pad // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(zp, win)[::hop]   # (n_frames, win)
    spec = np.fft.fftshift(np.fft.fft(frames * _HANN, axis=1), axes=1)  # частоты по возрастанию
    power_db = 10.0 * np.log10(np.abs(spec) ** 2 + _EPS)
    img = power_db.T[::-1]                      # частота по строкам, положительные сверху
    if img.shape[0] != config.IMG_SIZE:
        raise ValueError("STFT_WINDOW должен совпадать с IMG_SIZE")
    return img


def spectrogram(z):
    """Спектрограмма uint8 64x64 (см. spectrogram_db)."""
    return to_uint8(spectrogram_db(z))


# ---------------------------------------------------------------- скалограмма
_SCALES = np.geomspace(config.CWT_SCALE_MIN, config.CWT_SCALE_MAX, config.IMG_SIZE)


def scalogram_db(z):
    """Скалограмма в дБ (float, до перевода в uint8): CWT (комплексный Морле) отдельно для I и Q, мощности складываются.

    64 масштаба в геометрической прогрессии (малые масштабы = высокие частоты — сверху),
    усреднение по времени блоками до ширины 64, мощность в дБ.
    """
    z = normalize_power(_check_signal(z))
    iq = np.stack([z.real, z.imag])             # (2, N)
    coefs, _ = pywt.cwt(iq, _SCALES, config.CWT_WAVELET, method="fft", axis=-1)  # (S, 2, N)
    power = (np.abs(coefs) ** 2).sum(axis=1)    # (S, N): I и Q вместе
    n = power.shape[1]
    block = n // config.IMG_SIZE
    if block * config.IMG_SIZE != n:
        raise ValueError(f"длина сигнала {n} не делится на {config.IMG_SIZE}")
    power = power.reshape(power.shape[0], config.IMG_SIZE, block).mean(axis=2)
    return 10.0 * np.log10(power + _EPS)


def scalogram(z):
    """Скалограмма uint8 64x64 (см. scalogram_db)."""
    return to_uint8(scalogram_db(z))


# ---------------------------------------------------------------- созвездие
def _iq_histogram(points):
    """2D-гистограмма точек I/Q в квадрате [-R, R], log1p, uint8.

    Столбцы — I (слева направо), строки — Q (ось Q направлена вверх).
    """
    r = config.CONST_RANGE
    hist, _, _ = np.histogram2d(points.real, points.imag, bins=config.IMG_SIZE,
                                range=[[-r, r], [-r, r]])
    img = hist.T[::-1]                          # hist[i_bin, q_bin] -> строки Q сверху вниз
    return to_uint8(np.log1p(img))


def symbol_phase(z, sps=None):
    """Сдвиг k (0..sps-1), при котором отсчёты z[k::sps] имеют максимальную среднюю мощность.

    В максимуме мощности формирующего импульса отсчёт ближе всего к точке созвездия.
    """
    sps = config.SAMPLES_PER_SYMBOL if sps is None else sps
    n = len(z) // sps * sps
    power = (np.abs(z[:n]) ** 2).reshape(-1, sps).mean(axis=0)
    return int(np.argmax(power))


def constellation(z):
    """Диаграмма созвездия: по одному отсчёту на символ.

    После нормировки мощности берутся отсчёты z[k::SAMPLES_PER_SYMBOL] со сдвигом k
    максимальной средней мощности (1024 / 8 = 128 точек), строится гистограмма 64x64.
    """
    z = normalize_power(_check_signal(z))
    k = symbol_phase(z)
    return _iq_histogram(z[k::config.SAMPLES_PER_SYMBOL])


def iq_density(z):
    """Плотность I/Q: гистограмма по всем отсчётам сигнала (вместе с переходами между символами)."""
    return _iq_histogram(normalize_power(_check_signal(z)))


# ---------------------------------------------------------------- реестр
TRANSFORMS = {
    "spectrogram": spectrogram,
    "scalogram": scalogram,
    "constellation": constellation,
    "iq_density": iq_density,
}
assert tuple(TRANSFORMS) == config.ALL_REPRESENTATIONS


def all_representations(z, names=None):
    """Представления одного сигнала: словарь {имя: uint8 (64, 64)}.

    По умолчанию — все, включая вспомогательные (config.ALL_REPRESENTATIONS).
    """
    names = config.ALL_REPRESENTATIONS if names is None else names
    return {name: TRANSFORMS[name](z) for name in names}
