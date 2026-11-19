"""Единственное место с настройками проекта: пути, классы, параметры разбиения и преобразований."""

from pathlib import Path

# ---------------------------------------------------------------- пути
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FILE = RAW_DIR / "GOLD_XYZ_OSC.0001_1024.hdf5"
CACHE_DIR = DATA_DIR / "cache"
SPLIT_FILE = DATA_DIR / "splits" / "split.npz"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- датасет
# Исправленный порядок классов (classes-fixed.json), а не classes.txt.
CLASSES = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
)
NUM_CLASSES = len(CLASSES)

SNRS = tuple(range(-20, 31, 2))  # 26 уровней, дБ
SIGNAL_LEN = 1024

# ---------------------------------------------------------------- разбиение
SAMPLES_PER_CELL = 256              # сигналов на пару (класс, SNR)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42

# ---------------------------------------------------------------- изображения
IMG_SIZE = 64
REPRESENTATIONS = ("spectrogram", "scalogram", "constellation")  # основные
EXTRA_REPRESENTATIONS = ("iq_density",)                          # вспомогательные, для сравнения
ALL_REPRESENTATIONS = REPRESENTATIONS + EXTRA_REPRESENTATIONS

# Спектрограмма
STFT_WINDOW = 64
STFT_HOP = 16

# Скалограмма
CWT_WAVELET = "cmor1.5-1.0"
CWT_SCALE_MIN = 2.0
CWT_SCALE_MAX = 64.0

# Диаграмма созвездия и плотность I/Q: квадрат [-CONST_RANGE, CONST_RANGE]
CONST_RANGE = 2.5
# Отсчётов на символ в RadioML 2018.01A: созвездие строится по каждому 8-му отсчёту
SAMPLES_PER_SYMBOL = 8
