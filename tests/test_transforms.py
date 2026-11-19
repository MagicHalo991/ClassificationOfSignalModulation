"""Тесты преобразований на синтетических сигналах с известным правильным ответом.

Запуск из корня проекта:  py -m tests.test_transforms
Функции test_* совместимы с pytest, но pytest не обязателен.
"""

import sys
import time
import traceback

import numpy as np
import pywt
from scipy import ndimage

from source import config
from source.transforms import (TRANSFORMS, constellation, iq_density, normalize_power,
                               scalogram_db, spectrogram_db, symbol_phase)

N = config.SIGNAL_LEN
SPS = config.SAMPLES_PER_SYMBOL
RNG_SEED = 2026


# ---------------------------------------------------------------- генератор сигналов
def ideal_points(mod):
    """Идеальные точки созвездия со средней мощностью 1."""
    if mod == "BPSK":
        p = np.array([1, -1], complex)
    elif mod.endswith("PSK"):
        m = 4 if mod == "QPSK" else int(mod[:-3])
        p = np.exp(2j * np.pi * (np.arange(m) + 0.5 * (m == 4)) / m)
    elif mod.endswith("QAM"):
        m = int(np.sqrt(int(mod[:-3])))
        a = np.arange(-(m - 1), m, 2)
        p = (a[:, None] + 1j * a[None, :]).ravel()
    else:
        raise ValueError(mod)
    return p / np.sqrt(np.mean(np.abs(p) ** 2))


def pulse(kind, beta=0.35, span=8, sps=SPS):
    """Импульс: 'rc' — приподнятый косинус (без межсимвольных искажений), 'rrc' — корень из него."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    if kind == "rc":
        h = np.sinc(t) * np.cos(np.pi * beta * t)
        den = 1 - (2 * beta * t) ** 2
        sing = np.isclose(den, 0)
        h[~sing] /= den[~sing]
        h[sing] = np.pi / 4 * np.sinc(1 / (2 * beta))
    elif kind == "rrc":
        h = np.empty_like(t)
        for i, ti in enumerate(t):
            if np.isclose(ti, 0):
                h[i] = 1 + beta * (4 / np.pi - 1)
            elif np.isclose(abs(ti), 1 / (4 * beta)):
                h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            else:
                h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                       / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    else:
        raise ValueError(kind)
    return h


def modulate(mod, rng, kind="rc", snr_db=None, phase=None):
    """Цифровой сигнал длиной N с 8 отсчётами на символ.

    Возвращает (z, истинный сдвиг символьных отсчётов 0..7). SNR — отношение средней
    мощности сигнала к мощности шума на отсчёт.
    """
    h = pulse(kind)
    n_sym = N // SPS + len(h) // SPS + 2
    sym = rng.choice(ideal_points(mod), n_sym)
    up = np.zeros(n_sym * SPS, complex)
    up[::SPS] = sym
    full = np.convolve(up, h)
    shift = int(rng.integers(SPS)) if phase is None else (len(h) // 2 - phase) % SPS
    start = len(h) + shift
    z = full[start:start + N]
    true_phase = (len(h) // 2 - start) % SPS
    return add_noise(z, snr_db, rng), true_phase


def add_noise(z, snr_db, rng):
    if snr_db is None:
        return z.astype(np.complex64)
    p = np.mean(np.abs(z) ** 2)
    sigma = np.sqrt(p / 10 ** (snr_db / 10) / 2)
    return (z + sigma * (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z)))).astype(np.complex64)


def tone(f, n=N):
    return np.exp(2j * np.pi * f * np.arange(n)).astype(np.complex64)


def evm_db(points, mod):
    """EVM (дБ) точек относительно идеального созвездия.

    Поворот подбирается перебором, затем комплексное усиление — методом наименьших
    квадратов по решениям (у 128 случайных символов QAM мощность не равна точно 1).
    """
    ref = ideal_points(mod)
    pts = np.asarray(points, complex) / np.sqrt(np.mean(np.abs(points) ** 2))
    sym_order = 2 if mod == "BPSK" else (4 if "QAM" in mod or mod == "QPSK" else int(mod[:-3]))

    def decide(q):
        return ref[np.argmin(np.abs(q[:, None] - ref[None, :]), axis=1)]

    best = np.inf
    for th in np.linspace(0, 2 * np.pi / sym_order, 181):
        q = pts * np.exp(1j * th)
        for _ in range(3):
            d = decide(q)
            q = q * np.vdot(q, d) / np.vdot(q, q)      # МНК-усиление
        d = decide(q)
        best = min(best, np.mean(np.abs(q - d) ** 2) / np.mean(np.abs(d) ** 2))
    return 10 * np.log10(max(best, 1e-15))


def spec_row(f):
    """Ожидаемая строка спектрограммы для частоты f (доля частоты дискретизации)."""
    return config.IMG_SIZE // 2 - 1 - int(round(f * config.IMG_SIZE))


# ---------------------------------------------------------------- общие свойства
def test_format_and_edge_cases():
    rng = np.random.default_rng(RNG_SEED)
    z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    for name, fn in TRANSFORMS.items():
        img = fn(z)
        assert img.dtype == np.uint8 and img.shape == (64, 64), name
        assert img.min() == 0 and img.max() == 255, name
        zero_img = fn(np.zeros(N, np.complex64))
        if name in ("constellation", "iq_density"):
            # все точки в нуле -> одна яркая точка в центре гистограммы
            assert (zero_img > 0).sum() == 1 and zero_img[31:33, 31:33].max() == 255, name
        else:
            assert zero_img.max() == 0, f"{name}: нулевой сигнал -> не нули"
        for bad in (np.zeros(N), np.zeros((2, N), np.complex64)):
            try:
                fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{name}: не отверг вход {bad.shape} {bad.dtype}")
    return ("все 4 представления: uint8 64x64, диапазон 0..255; нулевой сигнал: спектрограмма и "
            "скалограмма — нули, созвездие и iq_density — одна точка в центре")


def test_amplitude_invariance():
    rng = np.random.default_rng(RNG_SEED)
    z, _ = modulate("16QAM", rng, snr_db=15)
    diffs = {name: int(np.abs(fn(z).astype(int) - fn((z * 37.5).astype(np.complex64)).astype(int)).max())
             for name, fn in TRANSFORMS.items()}
    assert max(diffs.values()) <= 1, diffs
    return f"умножение сигнала на 37.5: макс. разница пикселей {diffs}"


def test_phase_rotation_invariance():
    """Постоянный поворот фазы не меняет спектрограмму и скалограмму (|I|^2+|Q|^2 инвариантна)."""
    rng = np.random.default_rng(RNG_SEED)
    z, _ = modulate("8PSK", rng, snr_db=20)
    zr = (z * np.exp(1j * 1.1)).astype(np.complex64)
    d_spec = np.abs(spectrogram_db(z) - spectrogram_db(zr)).max()
    d_scal = np.abs(scalogram_db(z) - scalogram_db(zr)).max()
    assert d_spec < 1e-3 and d_scal < 1e-3, (d_spec, d_scal)    # точность float32
    return f"поворот на 63°: разница спектрограмм {d_spec:.1e} дБ, скалограмм {d_scal:.1e} дБ"


# ---------------------------------------------------------------- спектрограмма
def test_spectrogram_frequency_axis():
    freqs = [-0.45, -0.3, -0.25, -0.125, -0.05, 0.0, 0.05, 0.125, 0.25, 0.3, 0.45]
    rows = [int(np.argmax(spectrogram_db(tone(f)).mean(axis=1))) for f in freqs]
    exp = [spec_row(f) for f in freqs]
    assert rows == exp, list(zip(freqs, rows, exp))
    return f"тоны {freqs} -> строки {rows} (совпадают с ожидаемыми, +частоты сверху)"


def test_spectrogram_time_axis():
    z = tone(0.1)
    z[:N // 2] = 0
    z = add_noise(z, 30, np.random.default_rng(RNG_SEED))
    col = spectrogram_db(z)[spec_row(0.1)]
    first, last = col[:28].mean(), col[36:].mean()
    assert last - first > 20, (first, last)
    return f"тон только во второй половине: строка тона слева {first:.1f} дБ, справа {last:.1f} дБ"


def test_spectrogram_bandwidth():
    """Ширина полосы цифрового сигнала по уровню -6 дБ ≈ символьная скорость (1/8 -> 8 бинов)."""
    rng = np.random.default_rng(RNG_SEED)
    widths = []
    for _ in range(10):
        z, _ = modulate("QPSK", rng, snr_db=30)
        prof = 10 * np.log10((10 ** (spectrogram_db(z) / 10)).mean(axis=1))
        widths.append(int((prof > prof.max() - 6).sum()))
    assert 7 <= np.median(widths) <= 10, widths
    return f"QPSK, 8 отсч./символ: ширина по -6 дБ {np.median(widths):.0f} бинов (ожидается ~8)"


def test_spectrogram_snr_estimate():
    """Оценка SNR по спектрограмме (сигнал над шумовым полом / шум) совпадает с заданным."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for snr in (-5, 0, 5, 10, 15, 20):
        est = []
        for _ in range(8):
            z, _ = modulate("QPSK", rng, snr_db=snr)
            p = (10 ** (spectrogram_db(z) / 10)).mean(axis=1)       # средняя мощность по бинам
            out = np.r_[p[:16], p[-16:]]                              # |f| > 0.25 — вне полосы
            floor = out.mean()
            est.append(10 * np.log10(max(p.sum() - floor * 64, 1e-12) / (floor * 64)))
        rows.append((snr, float(np.mean(est))))
    err = max(abs(e - s) for s, e in rows)
    assert err < 1.5, rows
    return "заданный SNR -> оценка: " + ", ".join(f"{s:+d}->{e:+.1f}" for s, e in rows) + f" дБ (ошибка ≤ {err:.2f})"


def test_spectrogram_noise_flat():
    rng = np.random.default_rng(RNG_SEED)
    profs = []
    for _ in range(20):
        z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        profs.append(10 * np.log10((10 ** (spectrogram_db(z) / 10)).mean(axis=1)))
    prof = np.mean(profs, axis=0)
    assert prof.std() < 0.5, prof.std()
    return f"белый шум: неравномерность среднего спектра {prof.std():.2f} дБ (плоский)"


def test_spectrogram_edge_columns():
    """Известный артефакт: дополнение отражением делает крайние столбцы ярче при высоком SNR."""
    rng = np.random.default_rng(RNG_SEED)
    excess = []
    for _ in range(20):
        z, _ = modulate("QPSK", rng, snr_db=30)
        s = spectrogram_db(z)[:16]                  # вне полосы сигнала
        excess.append(s[:, [0, 63]].mean() - s[:, 8:56].mean())
    return f"ИНФО: крайние столбцы вне полосы ярче середины на {np.mean(excess):.1f} дБ при SNR 30 дБ"


# ---------------------------------------------------------------- скалограмма
def _scalo_expected_row(f):
    freqs = pywt.scale2frequency(config.CWT_WAVELET, np.geomspace(config.CWT_SCALE_MIN, config.CWT_SCALE_MAX,
                                                                  config.IMG_SIZE))
    return int(np.argmin(np.abs(np.log(freqs) - np.log(abs(f)))))


def test_scalogram_frequency_axis():
    freqs = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.45]
    rows = [int(np.argmax(scalogram_db(tone(f)).mean(axis=1))) for f in freqs]
    exp = [_scalo_expected_row(f) for f in freqs]
    assert all(abs(r - e) <= 1 for r, e in zip(rows, exp)), list(zip(freqs, rows, exp))
    assert rows == sorted(rows, reverse=True), "чем выше частота, тем выше строка"
    return f"тоны {freqs} -> строки {rows}, ожидались {exp} (±1)"


def test_scalogram_negative_frequency():
    """Отрицательная частота не теряется (I и Q считаются отдельно), но неотличима от положительной."""
    out = []
    for f in (0.05, 0.2):
        a, b = scalogram_db(tone(f)), scalogram_db(tone(-f))
        assert np.abs(a - b).max() < 1e-6 and a.max() > -10
        out.append(f)
    return f"тоны ±{out}: скалограммы совпадают — знак частоты скалограмма не различает"


def test_scalogram_time_axis():
    z = tone(0.1)
    z[:N // 2] = 0
    z = add_noise(z, 30, np.random.default_rng(RNG_SEED))
    row = scalogram_db(z)[_scalo_expected_row(0.1)]
    first, last = row[:28].mean(), row[36:].mean()
    assert last - first > 20, (first, last)
    return f"тон во второй половине: слева {first:.1f} дБ, справа {last:.1f} дБ"


def test_scalogram_digital_band():
    """Энергия цифрового сигнала (полоса ±1/16) сосредоточена на масштабах частот ≤ ~0.1."""
    rng = np.random.default_rng(RNG_SEED)
    z, _ = modulate("QPSK", rng, snr_db=30)
    prof = scalogram_db(z).mean(axis=1)
    peak_row = int(np.argmax(prof))
    f_peak = pywt.scale2frequency(config.CWT_WAVELET, np.geomspace(2, 64, 64)[peak_row])
    assert f_peak < 0.1, f_peak
    return f"QPSK: максимум скалограммы на частоте {f_peak:.3f} (строка {peak_row})"


# ---------------------------------------------------------------- созвездие
def test_constellation_axes():
    rng = np.random.default_rng(RNG_SEED)
    up = constellation((1j + 0.01 * rng.standard_normal(N)).astype(np.complex64))
    right = constellation((1 + 0.01j * rng.standard_normal(N)).astype(np.complex64))
    assert up[:32].sum() > 0 and up[32:].sum() == 0, "ось Q не вверх"
    assert right[:, 32:].sum() > 0 and right[:, :32].sum() == 0, "ось I не вправо"
    return "ось I вправо, ось Q вверх"


def test_constellation_noise_free():
    """Без шума и без межсимвольных искажений: верный сдвиг и точное созвездие."""
    rng = np.random.default_rng(RNG_SEED)
    res = []
    for mod in ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM"):
        hits, evms = 0, []
        for _ in range(20):
            z, ph = modulate(mod, rng, kind="rc")
            zn = normalize_power(z)
            k = symbol_phase(zn)
            hits += k == ph
            evms.append(evm_db(zn[k::SPS], mod))
        assert hits == 20 and max(evms) < -40, (mod, hits, max(evms))
        img = constellation(z)
        if mod in ("BPSK", "QPSK", "8PSK"):
            n_cl = ndimage.label(ndimage.binary_dilation(img > 0, iterations=1))[1]
            assert n_cl == {"BPSK": 2, "QPSK": 4, "8PSK": 8}[mod], (mod, n_cl)
        res.append(f"{mod}: сдвиг {hits}/20, EVM ≤ {max(evms):.0f} дБ")
    return "; ".join(res)


def test_constellation_awgn():
    """С шумом: доля верно выбранных сдвигов и EVM ≈ -SNR."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for snr in (0, 5, 10, 15, 20, 25, 30):
        exact, near, evms, loss = 0, 0, [], []
        for _ in range(30):
            z, ph = modulate("QPSK", rng, kind="rc", snr_db=snr)
            zn = normalize_power(z)
            k = symbol_phase(zn)
            dist = min((k - ph) % SPS, (ph - k) % SPS)
            exact += dist == 0
            near += dist <= 1
            e = evm_db(zn[k::SPS], "QPSK")
            evms.append(e)
            loss.append(e - evm_db(zn[ph::SPS], "QPSK"))   # проигрыш истинному сдвигу
        rows.append((snr, exact, near, float(np.median(evms)), float(np.mean(loss))))
    for snr, exact, near, e, lo in rows:
        if snr >= 10:
            # Известное ограничение: средняя мощность слабо зависит от сдвига (перепад ~10 %),
            # поэтому при 10..25 дБ иногда выбирается соседний сдвиг; проигрыш < 1 дБ в среднем.
            assert abs(-e - snr) < 2.5 and near >= 28 and lo < 1.5, rows
    return ("QPSK, SNR: сдвиг точно/±1 из 30, EVM, проигрыш истинному сдвигу — "
            + "; ".join(f"{s:+d} дБ: {x}/{n}, {e:+.1f} дБ, {lo:.2f} дБ" for s, x, n, e, lo in rows))


def test_constellation_tx_only_rrc():
    """ИНФО: только передающий RRC-фильтр (без согласованного) даёт межсимвольные искажения."""
    rng = np.random.default_rng(RNG_SEED)
    evms = [evm_db(normalize_power(z)[symbol_phase(normalize_power(z))::SPS], "QPSK")
            for z, _ in (modulate("QPSK", rng, kind="rrc") for _ in range(20))]
    return (f"ИНФО: QPSK с одним RRC-фильтром без шума: EVM {np.median(evms):.1f} дБ "
            f"({100 * 10 ** (np.median(evms) / 20):.0f} %) — точки размыты и без шума")


def test_iq_density_vs_constellation():
    rng = np.random.default_rng(RNG_SEED)
    z, _ = modulate("QPSK", rng, kind="rc")
    a, b = (constellation(z) > 0).sum(), (iq_density(z) > 0).sum()
    assert b > 3 * a, (a, b)
    return f"QPSK без шума: занято пикселей constellation {a}, iq_density {b} (видны переходы)"


def test_speed():
    rng = np.random.default_rng(RNG_SEED)
    z, _ = modulate("QPSK", rng, snr_db=10)
    out = []
    for name, fn in TRANSFORMS.items():
        fn(z)
        t0 = time.perf_counter()
        for _ in range(30):
            fn(z)
        out.append(f"{name} {(time.perf_counter() - t0) / 30 * 1e3:.2f} мс")
    return "время на сигнал: " + ", ".join(out)


# ---------------------------------------------------------------- запуск
def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            msg = fn()
            print(f"PASS  {name}\n      {msg}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\nИтого: {len(tests) - failed} из {len(tests)} прошли")
    return failed


if __name__ == "__main__":
    sys.exit(main())
