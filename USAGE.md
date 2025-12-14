# Руководство по использованию

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Базовое использование

Обработка одного изображения:
```bash
python floor_plan_parser.py --input plan.png
```

Обработка нескольких изображений:
```bash
python floor_plan_parser.py --input plan1.png plan2.jpg plan3.png --output results.json
```

### 3. Расширенные опции

#### Использование LSD детектора (более точный, но медленнее)
```bash
python floor_plan_parser.py --input plan.png --lsd
```

#### Детекция дверей и окон
```bash
python floor_plan_parser.py --input plan.png --detect-openings
```

#### Настройка минимальной длины линии
```bash
python floor_plan_parser.py --input plan.png --min-length 30
```

#### Комбинация опций
```bash
python floor_plan_parser.py --input plan1.png plan2.png --lsd --detect-openings --output full_results.json
```

## Программное использование

### Пример 1: Простая обработка

```python
from floor_plan_parser import FloorPlanParser

parser = FloorPlanParser()
result = parser.process_image("plan.png")
print(f"Найдено стен: {len(result['walls'])}")
```

### Пример 2: С детекцией отверстий

```python
from floor_plan_parser import FloorPlanParser

parser = FloorPlanParser(min_line_length=50)
result = parser.process_image("plan.png", detect_openings=True)

print(f"Стены: {len(result['walls'])}")
print(f"Отверстия: {len(result.get('openings', []))}")
```

### Пример 3: Пакетная обработка

```python
from floor_plan_parser import FloorPlanParser, save_results

parser = FloorPlanParser()
image_paths = ["plan1.png", "plan2.jpg", "plan3.png"]
results = parser.process_multiple_images(image_paths, detect_openings=True)
save_results(results, "batch_results.json")
```

## Настройка параметров

### Параметры конструктора FloorPlanParser

- `min_line_length` (по умолчанию 50): Минимальная длина линии в пикселях
- `max_line_gap` (по умолчанию 10): Максимальный разрыв между точками линии
- `canny_low` (по умолчанию 50): Нижний порог для Canny edge detection
- `canny_high` (по умолчанию 150): Верхний порог для Canny edge detection

Пример:
```python
parser = FloorPlanParser(
    min_line_length=30,  # Для более коротких линий
    max_line_gap=5,      # Более строгое объединение
    canny_low=30,        # Более чувствительное обнаружение краев
    canny_high=100
)
```

## Формат входных данных

### Поддерживаемые форматы
- PNG
- JPG/JPEG
- BMP
- TIFF

### Рекомендации по качеству изображений
- Четкие линии (не размытые)
- Хороший контраст между стенами и фоном
- Минимальный шум
- Разрешение не менее 800x600 пикселей

## Формат выходных данных

### Структура JSON

```json
[
  {
    "meta": {
      "source": "filename.png"
    },
    "walls": [
      {
        "id": "w1",
        "points": [[x1, y1], [x2, y2]]
      }
    ],
    "openings": [
      {
        "id": "o1",
        "type": "door",
        "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "bbox": {"x": 100, "y": 200, "width": 50, "height": 150},
        "area": 7500.0
      }
    ]
  }
]
```

### Описание полей

**walls:**
- `id`: Уникальный идентификатор стены
- `points`: Массив точек [[x1, y1], [x2, y2]] для линии или больше точек для полилинии

**openings** (опционально):
- `id`: Уникальный идентификатор отверстия
- `type`: Тип - "door", "window" или "opening"
- `points`: Массив точек контура
- `bbox`: Ограничивающий прямоугольник
- `area`: Площадь в пикселях

## Устранение проблем

### Проблема: Находится слишком много/мало линий

**Решение:**
- Увеличьте `min_line_length` для фильтрации коротких линий
- Настройте пороги Canny (`canny_low`, `canny_high`)
- Попробуйте использовать `--lsd` для более точной детекции

### Проблема: Линии разбиты на сегменты

**Решение:**
- Увеличьте `max_line_gap` в коде (сейчас жестко задано в `merge_nearby_lines`)
- Используйте `--lsd` который лучше справляется с непрерывными линиями

### Проблема: Не находятся стены на темном фоне

**Решение:**
- Алгоритм автоматически инвертирует изображение, если фон светлый
- Проверьте качество исходного изображения

### Проблема: Детекция отверстий работает плохо

**Решение:**
- Убедитесь, что отверстия четко видны на изображении
- Попробуйте предобработать изображение вручную (увеличить контраст)

## Визуализация результатов

После обработки изображений можно визуализировать результаты:

```bash
# Базовая визуализация
python visualize_results.py --json output.json --images-dir . --output-dir visualized

# Без отверстий
python visualize_results.py --json output.json --no-openings

# Настройка внешнего вида
python visualize_results.py --json output.json --wall-color 0 0 255 --wall-thickness 3
```

## Примеры команд

```bash
# Базовый пример
python floor_plan_parser.py -i test.png

# Полная обработка с отверстиями
python floor_plan_parser.py -i test.png --detect-openings --lsd -o result.json

# Обработка нескольких файлов
python floor_plan_parser.py -i plan1.png plan2.png plan3.png -o all_results.json

# Точная обработка с настройками
python floor_plan_parser.py -i test.png --lsd --min-length 30

# Визуализация результатов
python visualize_results.py --json all_results.json --images-dir . -o visualized
```


