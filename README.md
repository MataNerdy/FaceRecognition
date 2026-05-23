# Face Recognition Pipeline

End-to-end проект по распознаванию лиц на PyTorch: от поиска ключевых точек лица и выравнивания до обучения embedding-моделей, сравнения loss-функций и оценки качества через Identification Rate / TPR@FPR.

Проект собран из пяти исследовательских ноутбуков и оформлен в репозиторий с воспроизводимой структурой, скриптами, моделями и визуализациями.

![Pipeline example](assets/3_FaceRecognitionPipeline_Clean_cell18_out0.png)

---

## Что реализовано

### 1. Landmark detection и face alignment

Реализована и обучена **Stacked Hourglass Network** для предсказания 5 ключевых точек лица:

- левый глаз;
- правый глаз;
- нос;
- левый угол рта;
- правый угол рта.

На основе предсказанных landmarks реализовано выравнивание лица через affine/similarity transform. Это используется как второй этап классического face recognition pipeline.

Ключевые файлы:

- `src/face_recognition/models/hourglass.py`
- `src/face_recognition/alignment.py`
- `notebooks/1_StackedHourGlassNetwork_Clean.ipynb`

---

### 2. Обучение моделей распознавания лиц

Для распознавания лиц обучены модели на выровненных изображениях CelebA. В качестве backbone использовалась CNN-модель, предобученная на ImageNet, но не предобученная на face recognition.

Протестированы варианты:

| Подход | Что проверялось |
|---|---|
| CrossEntropy | базовая классификация identity |
| ArcFace | angular margin loss для более разделимых embedding-векторов |
| Triplet Loss | обучение embedding-пространства по тройкам anchor-positive-negative |
| CE + ArcFace | гибридный loss |
| ArcFace + Triplet | гибрид margin-based и metric learning подходов |

Ключевые файлы:

- `src/face_recognition/models/embedding.py`
- `src/face_recognition/losses.py`
- `src/face_recognition/datasets.py`
- `notebooks/2_AllModels_Clean (1).ipynb`

---

### 3. End-to-end pipeline

Собран полный pipeline:

1. найти лица на изображении;
2. получить landmarks;
3. выровнять найденные лица;
4. пропустить лица через embedding model;
5. сравнить embeddings через cosine similarity и L2 distance.

![Aligned faces](assets/3_FaceRecognitionPipeline_Clean_cell10_out0.png)

Ключевые файлы:

- `notebooks/3_FaceRecognitionPipeline_Clean.ipynb`
- `src/face_recognition/metrics.py`
- `src/face_recognition/utils/visualization.py`

---

### 4. Identification Rate Metric

Accuracy хорошо работает для закрытой классификации по известным identity, но плохо отражает качество модели на новых людях, которых не было в train/val.

Поэтому дополнительно реализована метрика **Identification Rate / TPR@FPR**:

- `query` содержит identity, не использовавшиеся при обучении;
- `distractors` содержит другие unseen identity;
- считаются cosine similarities для positive pairs, negative pairs и query-distractor pairs;
- по заданному FPR выбирается threshold;
- измеряется TPR.

Для оценки было подготовлено:

| Split | Количество лиц |
|---|---:|
| Query | 812 |
| Distractors | 617 |

Ключевой файл:

- `notebooks/4_Identification_Rate_Metric_Clean.ipynb`

---

## Лучший результат

По задаче **Identification Rate** лучшей оказалась модель, обученная на **CrossEntropy**.

| Model | TPR@FPR=0.50 | TPR@FPR=0.20 | TPR@FPR=0.10 | TPR@FPR=0.05 |
|---|---:|---:|---:|---:|
| CE | **0.8678** | **0.5805** | **0.3858** | **0.2431** |
| ArcFace | 0.6393 | 0.2936 | 0.1677 | 0.0976 |
| Triplet | 0.7215 | 0.3877 | 0.2285 | 0.1333 |
| CE + ArcFace | 0.7172 | 0.3668 | 0.2127 | 0.1282 |
| ArcFace + Triplet | 0.7192 | 0.3587 | 0.1991 | 0.1112 |

Вывод: в данном эксперименте CE оказался наиболее стабильным на unseen identity. Triplet Loss занял второе место, а ArcFace, вероятно, требовал большего числа identity и более масштабного обучения.

---

## Основные ML-инсайты

### CrossEntropy может быть сильным baseline

Хотя CE формально обучает классификатор по фиксированным identity, embedding backbone после такого обучения дал лучший результат на Identification Rate. Это важный baseline, который нельзя пропускать.

### ArcFace чувствителен к масштабу данных

ArcFace обычно силён в face recognition, но на ограниченном количестве identity может не раскрыться. В этом проекте он показал худший результат среди одиночных loss-функций.

### Triplet Loss лучше отражает задачу similarity search

Triplet Loss показал хорошее разделение лиц на внешних примерах: в top-k похожих пар вошли все 7/7 настоящих совпадений. Однако по строгому TPR@FPR он уступил CE.

### Accuracy недостаточно для face recognition

Для открытого множества людей нужна отдельная проверка на unseen identities. Поэтому Identification Rate / TPR@FPR важнее простой accuracy на validation split.

### End-to-end качество зависит от alignment

Ошибки на этапе landmarks и выравнивания напрямую портят embedding. Поэтому landmark detector — не декоративный этап, а важная часть pipeline.

---

## Сравнение open-source библиотек

Дополнительно протестированы библиотеки:

| Library | Результат |
|---|---|
| DeepFace | стабильно работает в Colab, поддерживает разные модели и backend |
| InsightFace | сильная библиотека для production/research face recognition, корректно работает через ONNX/GPU |
| face_recognition | проблемная совместимость с Colab из-за `dlib`, не рекомендована для этого окружения |

Ключевой файл:

- `notebooks/5_BiblioTest_Clean.ipynb`

---

## Структура репозитория

```text
face_recognition_portfolio/
├── assets/                         # графики и визуализации из ноутбуков
├── configs/                        # место для yaml/json конфигов экспериментов
├── notebooks/                      # исходные clean notebooks
├── scripts/
│   ├── train_hourglass.py          # entrypoint для landmark detector
│   ├── train_classifier.py         # entrypoint для classifier/embedding моделей
│   └── evaluate_ir.py              # сохраненные результаты IR metric
├── src/
│   └── face_recognition/
│       ├── alignment.py            # heatmaps -> landmarks -> aligned face
│       ├── datasets.py             # датасеты и transforms
│       ├── losses.py               # ArcFace, Triplet, hybrid losses
│       ├── metrics.py              # cosine similarity, embeddings, IR metric
│       ├── models/
│       │   ├── embedding.py        # ResNet embedding/classifier модели
│       │   └── hourglass.py        # Stacked Hourglass Network
│       └── utils/
│           └── visualization.py    # визуализация лиц и метрик
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Как запустить

Установить зависимости:

```bash
pip install -r requirements.txt
pip install -e .
```

Проверить сохраненные результаты IR metric:

```bash
python scripts/evaluate_ir.py
```

Запустить skeleton для модели landmarks:

```bash
python scripts/train_hourglass.py --checkpoint-out hourglass_best.pth
```

Запустить skeleton для classifier-модели:

```bash
python scripts/train_classifier.py --data-dir path/to/aligned_classifier_train_500
```

Данные CelebA и веса моделей не включены в репозиторий, потому что они тяжелые и должны храниться отдельно в `data/`, `datasets/` или `checkpoints/`.

---

## Используемые технологии

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- CelebA
- Stacked Hourglass Network
- ResNet backbone
- CrossEntropy Loss
- ArcFace Loss
- Triplet Loss
- Cosine Similarity
- TPR@FPR / Identification Rate
- DeepFace
- InsightFace

---

## STAR

### Situation

Face recognition pipeline состоит из нескольких зависимых этапов: детекция лица, локализация ключевых точек, выравнивание, построение embedding-вектора и сравнение лиц. Ошибка на любом этапе ухудшает итоговое качество.

Кроме того, обычная validation accuracy не показывает, как модель работает на новых людях, которых не было в обучении.

### Task

Нужно было реализовать полный pipeline распознавания лиц без использования моделей, заранее обученных на face recognition:

- обучить landmark detector;
- реализовать face alignment;
- обучить модели распознавания лиц с CE и ArcFace;
- дополнительно проверить Triplet Loss и гибридные loss-функции;
- собрать inference pipeline;
- реализовать Identification Rate Metric;
- сравнить результат с open-source библиотеками.

### Action

Была реализована Stacked Hourglass Network для 5 landmarks, подготовлены aligned face datasets, обучены embedding/classification модели на CE, ArcFace, Triplet и гибридных loss-функциях.

Для честной оценки были сформированы отдельные query/distractor наборы из identity, отсутствующих в train/val. Затем реализован расчет TPR@FPR по cosine similarities.

### Result

Лучшей моделью по Identification Rate стала CE-модель:

- TPR@FPR=0.50: **0.8678**
- TPR@FPR=0.20: **0.5805**
- TPR@FPR=0.10: **0.3858**
- TPR@FPR=0.05: **0.2431**

Triplet Loss показал сильное поведение на similarity-парах, но уступил CE по строгой IR-метрике. ArcFace оказался чувствителен к масштабу данных и в текущем сетапе не превзошел baseline.
