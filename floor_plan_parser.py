"""
Floor Plan Parser - Извлечение геометрии стен из изображений планов
Использует гибридный подход: OpenCV для предобработки и детекции линий
"""

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from pathlib import Path


class FloorPlanParser:
    """
    Парсер планов этажей для извлечения геометрии стен.
    
    Подход:
    - OpenCV для предобработки (бинаризация, морфология)
    - Детекция линий через HoughLinesP и LSD (Line Segment Detector)
    - Группировка и фильтрация линий
    - Постобработка для объединения близких сегментов
    """
    
    def __init__(self, 
                 min_line_length: int = 50,
                 max_line_gap: int = 10,
                 canny_low: int = 50,
                 canny_high: int = 150):
        """
        Args:
            min_line_length: Минимальная длина линии для детекции
            max_line_gap: Максимальный разрыв между точками линии
            canny_low: Нижний порог для Canny edge detection
            canny_high: Верхний порог для Canny edge detection
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.canny_low = canny_low
        self.canny_high = canny_high
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для улучшения детекции линий.
        
        Этапы:
        1. Конвертация в grayscale
        2. Бинаризация (Otsu или адаптивная)
        3. Морфологические операции для очистки
        4. Инверсия (если нужно)
        """
        # Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Бинаризация с Otsu для автоматического выбора порога
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Инверсия, если фон светлый (стены обычно темные на планах)
        # Проверяем среднюю яркость
        mean_brightness = np.mean(binary)
        if mean_brightness > 127:
            binary = cv2.bitwise_not(binary)
        
        # Морфологические операции для очистки шума
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def detect_lines_hough(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детекция линий через Probabilistic Hough Transform.
        Хорошо работает для прямых линий в архитектурных планах.
        """
        # Edge detection
        edges = cv2.Canny(binary, self.canny_low, self.canny_high)
        
        # HoughLinesP - вероятностный вариант, быстрее и точнее
        lines = cv2.HoughLinesP(
            edges,
            rho=1,              # Разрешение по rho (пиксели)
            theta=np.pi/180,    # Разрешение по theta (радианы)
            threshold=80,       # Минимальное количество пересечений
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        return [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]
    
    def detect_lines_lsd(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детекция линий через Line Segment Detector (LSD).
        Более точный, но медленнее. Хорош для тонких линий.
        """
        # Создаем LSD детектор
        lsd = cv2.createLineSegmentDetector(0)
        
        # Детекция линий
        lines, _, _, _ = lsd.detect(binary)
        
        if lines is None:
            return []
        
        # Конвертация в формат (x1, y1, x2, y2)
        result = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            result.append((int(x1), int(y1), int(x2), int(y2)))
        
        return result
    
    def merge_nearby_lines(self, lines: List[Tuple[int, int, int, int]], 
                          distance_threshold: float = 10.0,
                          angle_threshold: float = 5.0) -> List[Tuple[int, int, int, int]]:
        """
        Объединение близких и коллинеарных линий.
        Уменьшает количество дублирующихся сегментов.
        """
        if not lines:
            return []
        
        merged = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            x1, y1, x2, y2 = line1
            current_group = [line1]
            used[i] = True
            
            # Вычисляем угол и длину первой линии
            angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            for j, line2 in enumerate(lines[i+1:], start=i+1):
                if used[j]:
                    continue
                
                x3, y3, x4, y4 = line2
                angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
                
                # Проверяем угол (учитываем периодичность 180 градусов)
                angle_diff = abs(angle1 - angle2)
                angle_diff = min(angle_diff, 180 - angle_diff)
                
                if angle_diff > angle_threshold:
                    continue
                
                # Проверяем расстояние между линиями
                # Расстояние от точки до линии
                def point_to_line_dist(px, py, x1, y1, x2, y2):
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - x1 * y2
                    return abs(A * px + B * py + C) / np.sqrt(A * A + B * B) if (A * A + B * B) > 0 else float('inf')
                
                dist1 = point_to_line_dist(x3, y3, x1, y1, x2, y2)
                dist2 = point_to_line_dist(x4, y4, x1, y1, x2, y2)
                
                if min(dist1, dist2) <= distance_threshold:
                    current_group.append(line2)
                    used[j] = True
            
            # Объединяем линии в группе
            if len(current_group) > 1:
                # Находим крайние точки
                all_points = []
                for line in current_group:
                    all_points.extend([(line[0], line[1]), (line[2], line[3])])
                
                # Сортируем точки по проекции на направление линии
                if len(all_points) > 0:
                    # Используем первую линию как направление
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        continue
                    
                    # Проекция точек на направление
                    projections = []
                    for px, py in all_points:
                        proj = (px - x1) * dx + (py - y1) * dy
                        projections.append((proj, px, py))
                    
                    projections.sort()
                    merged.append((projections[0][1], projections[0][2], 
                                  projections[-1][1], projections[-1][2]))
            else:
                merged.append(line1)
        
        return merged
    
    def filter_lines_by_length(self, lines: List[Tuple[int, int, int, int]], 
                              min_length: int = 30) -> List[Tuple[int, int, int, int]]:
        """Фильтрация линий по минимальной длине."""
        filtered = []
        for x1, y1, x2, y2 in lines:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= min_length:
                filtered.append((x1, y1, x2, y2))
        return filtered
    
    def detect_walls(self, image_path: str, use_lsd: bool = False) -> List[Dict]:
        """
        Основной метод детекции стен из изображения.
        
        Args:
            image_path: Путь к изображению
            use_lsd: Использовать LSD вместо Hough (медленнее, но точнее)
        
        Returns:
            Список словарей с информацией о стенах
        """
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Предобработка
        binary = self.preprocess_image(image)
        
        # Детекция линий
        if use_lsd:
            lines = self.detect_lines_lsd(binary)
        else:
            lines = self.detect_lines_hough(binary)
        
        # Фильтрация по длине
        lines = self.filter_lines_by_length(lines, min_length=self.min_line_length)
        
        # Объединение близких линий
        lines = self.merge_nearby_lines(lines, distance_threshold=15.0, angle_threshold=3.0)
        
        # Конвертация в формат для JSON
        walls = []
        for i, (x1, y1, x2, y2) in enumerate(lines, start=1):
            walls.append({
                "id": f"w{i}",
                "points": [[int(x1), int(y1)], [int(x2), int(y2)]]
            })
        
        return walls
    
    def detect_openings(self, image_path: str, walls: List[Dict]) -> List[Dict]:
        """
        Детекция дверей и окон (отверстий в стенах).
        
        Подход:
        1. Находим разрывы в стенах (пробелы между сегментами)
        2. Детекция прямоугольных объектов определенного размера
        3. Анализ контуров для поиска характерных форм
        
        Args:
            image_path: Путь к изображению
            walls: Список найденных стен
        
        Returns:
            Список словарей с информацией об отверстиях
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        binary = self.preprocess_image(image)
        
        # Инвертируем для поиска светлых областей (отверстия обычно светлее)
        binary_inv = cv2.bitwise_not(binary)
        
        # Морфологические операции для выделения прямоугольных объектов
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_rect)
        
        # Поиск контуров
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        openings = []
        opening_id = 1
        
        for contour in contours:
            # Вычисляем площадь и периметр
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Фильтруем по размеру (двери/окна имеют характерные размеры)
            # Минимальная площадь для двери ~500 пикселей, для окна ~200
            if area < 100 or area > 10000:
                continue
            
            # Аппроксимация контура
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ищем прямоугольные формы (4 вершины)
            if len(approx) >= 4:
                # Вычисляем bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Проверяем соотношение сторон (двери обычно вертикальные, окна горизонтальные)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Классификация по размеру и форме
                opening_type = "opening"  # По умолчанию
                if aspect_ratio < 0.5 and h > w * 2:
                    opening_type = "door"  # Вертикальный прямоугольник
                elif aspect_ratio > 2 and w > h * 2:
                    opening_type = "window"  # Горизонтальный прямоугольник
                elif 0.7 < aspect_ratio < 1.4:
                    # Квадратные формы - могут быть окнами
                    if area < 2000:
                        opening_type = "window"
                    else:
                        opening_type = "door"
                
                # Получаем точки контура
                points = [[int(point[0][0]), int(point[0][1])] for point in approx]
                
                openings.append({
                    "id": f"o{opening_id}",
                    "type": opening_type,
                    "points": points,
                    "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "area": float(area)
                })
                opening_id += 1
        
        return openings
    
    def process_image(self, image_path: str, use_lsd: bool = False, 
                     detect_openings: bool = False) -> Dict:
        """
        Обработка одного изображения и возврат JSON структуры.
        
        Args:
            image_path: Путь к изображению
            use_lsd: Использовать LSD детектор
            detect_openings: Детектировать двери и окна
        
        Returns:
            Словарь с метаданными и стенами
        """
        filename = os.path.basename(image_path)
        walls = self.detect_walls(image_path, use_lsd=use_lsd)
        
        result = {
            "meta": {
                "source": filename
            },
            "walls": walls
        }
        
        # Детекция отверстий (двери/окна)
        if detect_openings:
            openings = self.detect_openings(image_path, walls)
            result["openings"] = openings
        
        return result
    
    def process_multiple_images(self, image_paths: List[str], 
                               use_lsd: bool = False,
                               detect_openings: bool = False) -> List[Dict]:
        """
        Обработка нескольких изображений.
        
        Args:
            image_paths: Список путей к изображениям
            use_lsd: Использовать LSD детектор
            detect_openings: Детектировать двери и окна
        
        Returns:
            Список результатов для каждого изображения
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, use_lsd=use_lsd, 
                                          detect_openings=detect_openings)
                results.append(result)
                walls_count = len(result.get('walls', []))
                openings_count = len(result.get('openings', []))
                msg = f"✓ Обработано: {os.path.basename(image_path)} - найдено {walls_count} стен"
                if openings_count > 0:
                    msg += f", {openings_count} отверстий"
                print(msg)
            except Exception as e:
                print(f"✗ Ошибка при обработке {image_path}: {e}")
                results.append({
                    "meta": {"source": os.path.basename(image_path)},
                    "walls": [],
                    "error": str(e)
                })
        
        return results


def save_results(results: List[Dict], output_path: str = "output.json"):
    """Сохранение результатов в JSON файл."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Парсер планов этажей")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                       help="Пути к изображениям планов")
    parser.add_argument("--output", "-o", default="output.json",
                       help="Путь к выходному JSON файлу")
    parser.add_argument("--lsd", action="store_true",
                       help="Использовать LSD детектор (медленнее, но точнее)")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Минимальная длина линии")
    parser.add_argument("--detect-openings", action="store_true",
                       help="Детектировать двери и окна")
    
    args = parser.parse_args()
    
    # Создание парсера
    floor_parser = FloorPlanParser(min_line_length=args.min_length)
    
    # Обработка изображений
    print("Начало обработки изображений...")
    results = floor_parser.process_multiple_images(args.input, use_lsd=args.lsd,
                                                  detect_openings=args.detect_openings)
    
    # Сохранение результатов
    save_results(results, args.output)
    
    print(f"\nОбработано изображений: {len(results)}")
    total_walls = sum(len(r.get('walls', [])) for r in results)
    print(f"Всего найдено стен: {total_walls}")

