"""Расчёт кэша изображений для всех сигналов разбиения.

Для каждого представления создаётся data/cache/<имя>.npy — uint8 (N, 64, 64),
строки в том же порядке, что split["indices"]. Представления считаются по очереди,
чтобы скорость каждого была видна отдельно.

Запуск из корня проекта:
    py -m scripts.build_cache                              # три основных представления
    py -m scripts.build_cache --rep iq_density             # вспомогательное представление
    py -m scripts.build_cache --rep scalogram --workers 8
    py -m scripts.build_cache --limit 2000                 # пробный запуск
При --limit обрабатываются первые N сигналов и результат пишется в
data/cache/<имя>_trial.npy, чтобы не испортить основной кэш.
"""

import argparse
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np

from source import config
from source.dataset import load_split, read_signals
from source.transforms import TRANSFORMS

READ_CHUNK = 2048   # сигналов за одно чтение из HDF5
TASK_SIZE = 64      # сигналов в одной задаче для воркера


def _compute(task):
    """Задача воркера: (имя представления, начальная позиция, сигналы) -> изображения."""
    rep, start, signals = task
    fn = TRANSFORMS[rep]
    return start, np.stack([fn(z) for z in signals])


def _warmup(_):
    """Пустая задача для прогрева пула: заодно считает по одному изображению каждого вида."""
    z = np.exp(2j * np.pi * 0.1 * np.arange(config.SIGNAL_LEN)).astype(np.complex64)
    for fn in TRANSFORMS.values():
        fn(z)
    time.sleep(0.05)


def _tasks(rep, h5, indices):
    """Читает сигналы порциями по READ_CHUNK и нарезает их на задачи."""
    for start in range(0, len(indices), READ_CHUNK):
        signals = read_signals(h5, indices[start:start + READ_CHUNK])
        for off in range(0, len(signals), TASK_SIZE):
            yield rep, start + off, signals[off:off + TASK_SIZE]


def build(rep, indices, out_path, pool, raw_file):
    """Считает одно представление и записывает его в out_path. Возвращает скорость (сигн./с)."""
    n = len(indices)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                    shape=(n, config.IMG_SIZE, config.IMG_SIZE))
    t0 = last = time.perf_counter()
    done = 0
    with h5py.File(raw_file, "r") as h5:
        for start, images in pool.imap(_compute, _tasks(rep, h5, indices)):
            out[start:start + len(images)] = images
            done += len(images)
            now = time.perf_counter()
            if now - last >= 5.0 or done == n:
                speed = done / (now - t0)
                eta = (n - done) / speed if speed > 0 else float("inf")
                print(f"  {rep}: {done}/{n} ({100 * done / n:5.1f}%), "
                      f"{speed:7.1f} сигн./с, осталось ~{eta / 60:.1f} мин", flush=True)
                last = now
    out.flush()
    del out
    elapsed = time.perf_counter() - t0
    return n / elapsed if elapsed > 0 else float("inf")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Расчёт кэша изображений")
    parser.add_argument("--rep", nargs="+", choices=config.ALL_REPRESENTATIONS,
                        default=list(config.REPRESENTATIONS),
                        help="одно или несколько представлений (по умолчанию три основных)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="число процессов")
    parser.add_argument("--limit", type=int, default=None, help="обработать только первые N сигналов")
    args = parser.parse_args(argv)

    if not config.SPLIT_FILE.exists():
        raise SystemExit(f"нет {config.SPLIT_FILE}; сначала запустите py -m scripts.make_split")
    indices = load_split(config.SPLIT_FILE)["indices"]
    suffix = ""
    if args.limit is not None:
        indices = indices[:args.limit]
        suffix = "_trial"
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Сигналов: {len(indices)}, представления: {', '.join(args.rep)}, процессов: {args.workers}")
    speeds = {}
    with Pool(args.workers) as pool:
        # Прогрев: дождаться запуска всех процессов, чтобы старт не попал в замер скорости
        pool.map(_warmup, range(args.workers * 4), chunksize=1)
        for rep in args.rep:
            out_path = config.CACHE_DIR / f"{rep}{suffix}.npy"
            speeds[rep] = build(rep, indices, out_path, pool, config.RAW_FILE)
            print(f"  -> {out_path}")

    full = config.NUM_CLASSES * len(config.SNRS) * config.SAMPLES_PER_CELL
    print("\nИтог:")
    for rep, s in speeds.items():
        print(f"  {rep:14s} {s:8.1f} сигн./с; полный кэш ({full} сигн.) ~{full / s / 60:.1f} мин")
    return speeds


if __name__ == "__main__":
    main()
