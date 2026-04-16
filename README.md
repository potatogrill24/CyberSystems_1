# Лабораторная работа: Семантическая сегментация листьев клубники

## Описание работы

В работе исследуются модели семантической сегментации для обнаружения заболеваний листьев клубники. Цель — автоматическое выделение пораженных участков на листьях для диагностики болезней.

**Датасет:** Strawberry Disease Detection (Kaggle) — изображения листьев с полигональной разметкой пораженных областей в формате LabelMe.

## Выполненные задачи

- Подготовка данных и реализация кастомного Dataset для загрузки изображений и JSON-аннотаций
- Обучение базовых моделей из `segmentation_models_pytorch` (U-Net, DeepLabV3+)
- Проверка гипотез улучшения бейзлайна (аугментации, увеличение разрешения, CosineAnnealingLR)
- Самостоятельная реализация двух архитектур (MiniUNet и MiniDeepLabLike)
- Обучение собственных моделей на базовой и улучшенной конфигурациях
- Сравнение моделей по метрикам mIoU, Dice Score, Pixel Accuracy

## Установка и запуск в Google Colab

1. Открыть ноутбук в Colab и включить GPU: `Среда выполнения` → `Сменить среду выполнения` → `T4 GPU`
2. Выполнить ячейки последовательно

## Устанока и запуск на локальной машине

1. Клонировать репозиторий:
```bash
git clone https://github.com/potatogrill24/CyberSystems_1/edit/main
cd CyberSystems_1
```
2. Подготовить виртуальное окружение. Для windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
Для Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Установить зависимости:
```bash
pip install --upgrade pip
pip uninstall -y segmentation-models-pytorch timm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade timm segmentation-models-pytorch
pip install albumentations opencv-python matplotlib pandas tqdm jupyter
```
4. Выполнить все ячейки последовательно

## Основные результаты

| Модель | Конфигурация | mIoU | Dice |
|--------|-------------|------|------|
| DeepLabV3+ (SMP) | Baseline | 0.7379 | 0.8423 |
| DeepLabV3+ (SMP) | Улучшенный (H2) | **0.7433** | **0.8463** |
| MiniDeepLabLike | Baseline | 0.5481 | 0.6818 |
| MiniDeepLabLike | Улучшенный (H2) | 0.5103 | 0.6471 |
| MiniUNet | Baseline | 0.4050 | 0.5112 |
| MiniUNet | Улучшенный (H2) | 0.3105 | 0.3790 |

## Выводы

Лучший результат показала DeepLabV3+ с улучшенной конфигурацией H2 (mIoU = 0.7433). Библиотечные модели значительно превосходят самописные благодаря предобученным энкодерам. Улучшения, работающие для крупных моделей, оказались слишком агрессивными для легковесных архитектур.
