# External Libraries Baseline

В пятом исследовательском ноутбуке были проверены готовые open-source библиотеки для face recognition: DeepFace, InsightFace и `face_recognition` на базе dlib. Это сравнение не является частью основного pipeline, а служит sanity-check и исследовательским baseline: важно понимать, как ведут себя готовые решения и какие trade-off возникают при их использовании.

## Что проверялось

- **DeepFace**: удобный high-level API, поддержка нескольких моделей, включая ArcFace, простое извлечение лиц и embeddings.
- **InsightFace**: сильная research/production библиотека с ONNX Runtime, готовым alignment и качественными ArcFace-like моделями.
- **face_recognition / dlib**: простой API, но в Colab возникли проблемы совместимости и производительности из-за dlib.

## Почему они не используются в основном pipeline

Цель проекта учебно-исследовательская: реализовать собственный путь от landmarks и alignment до обучения classifier/embedding моделей. Поэтому pretrained face-recognition модели из DeepFace или InsightFace не используются для финальных метрик проекта. ImageNet initialization допустим только как общий CNN backbone, не как pretrained face-recognition encoder.

## Вывод

DeepFace и InsightFace полезны как baseline и источник инженерных идей. `face_recognition` удобен для небольших задач, но в данном окружении менее стабилен из-за зависимости от dlib. Основной репозиторий сохраняет эти эксперименты в notebook-only формате, чтобы не перегружать воспроизводимый pipeline тяжелыми внешними runtime-зависимостями.
