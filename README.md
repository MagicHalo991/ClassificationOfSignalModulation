# ClassificationOfSignalModulation

## Назначение

Учебный проект: сравнение трёх визуальных представлений радиосигнала —
**спектрограммы**, **скалограммы** и **диаграммы созвездия** — для автоматической
классификации видов модуляции свёрточными нейросетями на датасете
**RadioML 2018.01A** (24 вида модуляции, SNR от −20 до +30 дБ).

**Срок сдачи: 25 декабря 2026 г.**

## Установка

Windows 11, PowerShell, Python 3.13 (запускается командой `py`).

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

После активации окружения `py` и `python` указывают на `.venv`.

## Датасет

Нужен файл `GOLD_XYZ_OSC.0001_1024.hdf5` (21,4 ГБ) из RadioML 2018.01A.
Положите его в `data/raw/` вместе с `classes-fixed.json`:

```
data/raw/GOLD_XYZ_OSC.0001_1024.hdf5
data/raw/classes-fixed.json
data/raw/LICENSE.TXT
```

Скачать можно с Kaggle (датасет `pinxau1000/radioml2018`), нужен токен в `~/.kaggle/`:

```powershell
kaggle datasets download pinxau1000/radioml2018 -p data/raw
```

затем распаковать архив в `data/raw/` и удалить его. Для работы нужно около 45 ГБ
свободного места (архив + распакованный файл). Содержимое `data/raw/` и `data/cache/`
в git не попадает.

Внимание: порядок классов — по `classes-fixed.json`, а не по `classes.txt`
(в `classes.txt` он неверный). Список классов задан в `source/config.py`.

## Запуск скриптов

Все скрипты запускаются из корня проекта как модули:

```powershell
py -m scripts.check_dataset                 # формы массивов, SNR, число сигналов; results/classes_check.png
py -m scripts.make_split                    # разбиение train/val/test -> data/splits/split.npz
py -m scripts.build_cache --limit 2000      # пробный расчёт (пишет data/cache/<имя>_trial.npy)
py -m scripts.build_cache                   # полный кэш трёх основных представлений
py -m scripts.build_cache --rep iq_density  # вспомогательное представление
py -m scripts.build_cache --rep spectrogram scalogram constellation iq_density   # все четыре
py -m scripts.build_cache --rep scalogram --workers 8
py -m scripts.show_representations          # results/representations.png и representations_snr.png
py -m scripts.validate_representations      # проверка на настоящих данных -> results/validation/
py -m scripts.save_docs_images              # иллюстрации для документации -> docs/images/
```

## Тесты

```powershell
py -m tests.test_transforms                 # 19 тестов преобразований на синтетических сигналах
```

Тесты не требуют датасета: генерируют сигналы с известной частотой, модуляцией, тактовым
сдвигом и SNR и проверяют оси, ширину полосы, оценку SNR, выбор отсчётов созвездия и т. д.
Результаты тестирования и проверки на настоящих данных — в [docs/testing.md](docs/testing.md).

Подвыборка: по `SAMPLES_PER_CELL = 256` сигналов на каждую пару (класс, SNR),
то есть 24 × 26 × 256 = 159 744 сигнала, деление 70/15/15 внутри каждой пары
(179 / 38 / 39 на пару). Кэш одного представления — uint8 (159 744, 64, 64), 654 МБ;
все четыре считаются примерно за 2 минуты на 16 ядрах (дольше всего скалограмма).

## Представления

Все изображения 64×64, uint8, мощность сигнала перед преобразованием нормируется к 1.
Основные представления (`config.REPRESENTATIONS`) — первые три; `iq_density` —
вспомогательное (`config.EXTRA_REPRESENTATIONS`), для сравнения с `constellation`.

| Представление | Как считается |
|---|---|
| `spectrogram`   | STFT: окно Ханна 64, шаг 16, отражение по краям до 64 окон; мощность в дБ; частота по вертикали (положительные сверху), время по горизонтали |
| `scalogram`     | CWT `cmor1.5-1.0`, 64 масштаба 2…64 (геометрически), отдельно I и Q, мощности складываются; усреднение по времени до 64 столбцов; дБ; мелкие масштабы (высокие частоты) сверху |
| `constellation` | Один отсчёт на символ: из `z[k::8]` (`SAMPLES_PER_SYMBOL = 8`) берётся сдвиг k с максимальной средней мощностью, 128 точек; 2D-гистограмма I/Q в квадрате [−2.5, 2.5], 64 бина, `log1p`; ось Q вверх |
| `iq_density`    | Та же гистограмма, но по всем 1024 отсчётам (вместе с переходами между символами) |

`results/classes_check.png` — созвездия всех 24 классов при +30 дБ;
`results/representations.png` — 6 классов × 4 представления при +30 дБ;
`results/representations_snr.png` — QPSK при −10…+30 дБ × 4 представления.

## Структура

```
main.py               точка входа приложения (пока пусто)
requirements.txt      зависимости
source/               библиотечный код
  config.py           ВСЕ настройки: пути, классы, SNR, разбиение, параметры преобразований
  transforms.py       сигнал -> изображения 64x64 (спектрограмма, скалограмма, созвездие, плотность I/Q)
  dataset.py          чтение HDF5, разбиение по (класс, SNR), ImageDataset для DataLoader
  models.py, train.py, evaluate.py, generator.py   — заготовки (пока пусто)
scripts/              запускаемые скрипты (py -m scripts.<имя>)
  check_dataset.py    проверка файла датасета
  make_split.py       построение разбиения
  build_cache.py      расчёт кэша изображений
  show_representations.py   обзорные картинки представлений
  validate_representations.py   проверка представлений на настоящих данных
  save_docs_images.py       иллюстрации в docs/images
  run_train.py, run_eval.py                        — заготовки (пока пусто)
ui/                   графическое приложение (PyQt6, позже)
notebooks/            исследовательские блокноты
data/raw/             исходный HDF5 (не в git)
data/cache/           кэш изображений <представление>.npy (не в git)
data/splits/          split.npz — номера сигналов и части train/val/test
checkpoints/          веса моделей (не в git)
tests/                тесты (py -m tests.test_transforms)
results/              графики и таблицы результатов (validation/ — проверка представлений)
docs/                 документация: testing.md — отчёт о тестировании, images/ — иллюстрации
resources/            ресурсы приложения
```
