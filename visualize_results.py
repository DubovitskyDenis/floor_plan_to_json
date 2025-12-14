"""
Скрипт для визуализации результатов парсинга планов этажей.
Отображает найденные стены и отверстия поверх исходного изображения.
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class FloorPlanVisualizer:
    """
    Визуализатор результатов парсинга планов этажей.
    """
    
    def __init__(self, 
                 wall_color: Tuple[int, int, int] = (0, 255, 0),
                 opening_color: Tuple[int, int, int] = (255, 0, 0),
                 wall_thickness: int = 2,
                 opening_thickness: int = 2,
                 show_labels: bool = True,
                 fill_walls: bool = True,
                 wall_fill_opacity: float = 0.5):
        """
        Args:
            wall_color: Цвет для отрисовки стен (BGR)
            opening_color: Цвет для отрисовки отверстий (BGR)
            wall_thickness: Толщина линий стен
            opening_thickness: Толщина линий отверстий
            show_labels: Показывать ли ID элементов
            fill_walls: Заполнять ли полигоны стен
            wall_fill_opacity: Прозрачность заливки стен (0.0 - полностью прозрачно, 1.0 - непрозрачно)
        """
        self.wall_color = wall_color
        self.opening_color = opening_color
        self.wall_thickness = wall_thickness
        self.opening_thickness = opening_thickness
        self.show_labels = show_labels
        self.fill_walls = fill_walls
        self.wall_fill_opacity = wall_fill_opacity
        
        # Цвета для разных типов отверстий
        self.opening_colors = {
            "door": (255, 0, 0),      # Синий
            "window": (0, 165, 255),  # Оранжевый
            "opening": (255, 255, 0)   # Голубой
        }
    
    def draw_wall(self, image: np.ndarray, wall: Dict, wall_id: str = None) -> np.ndarray:
        """
        Отрисовка одной стены на изображении.
        
        Args:
            image: Изображение для отрисовки
            wall: Словарь с данными стены {"id": "...", "points": [[x1,y1], [x2,y2], ...]}
            wall_id: ID стены (если не указан, берется из wall["id"])
        
        Returns:
            Изображение с нарисованной стеной
        """
        points = wall.get("points", [])
        if len(points) < 2:
            return image
        
        # Конвертация точек в формат для OpenCV
        pts = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
        
        # Отрисовка линии/полилинии
        if len(pts) == 2:
            # Для линии создаем заполненный прямоугольник
            if self.fill_walls:
                # Вычисляем перпендикулярный вектор для создания прямоугольника
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Нормализованный перпендикулярный вектор
                    perp_x = -dy / length
                    perp_y = dx / length
                    # Половина толщины
                    half_thick = self.wall_thickness / 2.0
                    
                    # Создаем 4 точки прямоугольника
                    rect_pts = np.array([
                        [int(pts[0][0] + perp_x * half_thick), int(pts[0][1] + perp_y * half_thick)],
                        [int(pts[1][0] + perp_x * half_thick), int(pts[1][1] + perp_y * half_thick)],
                        [int(pts[1][0] - perp_x * half_thick), int(pts[1][1] - perp_y * half_thick)],
                        [int(pts[0][0] - perp_x * half_thick), int(pts[0][1] - perp_y * half_thick)]
                    ], dtype=np.int32)
                    
                    # Создание overlay для заливки с прозрачностью
                    overlay = image.copy()
                    # Заливка прямоугольника
                    cv2.fillPoly(overlay, [rect_pts], self.wall_color)
                    # Наложение с прозрачностью
                    cv2.addWeighted(overlay, self.wall_fill_opacity, image, 
                                  1.0 - self.wall_fill_opacity, 0, image)
                    # Контур прямоугольника
                    cv2.polylines(image, [rect_pts], True, 
                                 self.wall_color, max(1, self.wall_thickness // 2))
                else:
                    # Если длина нулевая, просто рисуем точку
                    cv2.circle(image, tuple(pts[0]), self.wall_thickness, 
                              self.wall_color, -1)
            else:
                # Простая линия без заливки
                cv2.line(image, tuple(pts[0]), tuple(pts[1]), 
                        self.wall_color, self.wall_thickness)
        else:
            # Полигон с заливкой
            # Убеждаемся, что полигон замкнут (первая точка = последней)
            if not np.array_equal(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0]])
            
            if self.fill_walls:
                # Создание overlay для заливки с прозрачностью
                overlay = image.copy()
                # Заливка полигона
                cv2.fillPoly(overlay, [pts], self.wall_color)
                # Наложение с прозрачностью
                cv2.addWeighted(overlay, self.wall_fill_opacity, image, 
                              1.0 - self.wall_fill_opacity, 0, image)
            
            # Отрисовка контура полигона
            cv2.polylines(image, [pts], True, 
                         self.wall_color, self.wall_thickness)
        
        # Добавление ID метки
        if self.show_labels and wall_id:
            label = wall.get("id", wall_id)
            # Позиция метки - центр полигона
            if len(pts) > 2:
                # Центр масс для полигона
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    label_pos = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    label_pos = ((pts[0][0] + pts[-1][0]) // 2, 
                                (pts[0][1] + pts[-1][1]) // 2)
            else:
                # Середина линии
                label_pos = ((pts[0][0] + pts[-1][0]) // 2, 
                            (pts[0][1] + pts[-1][1]) // 2)
            cv2.putText(image, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.wall_color, 1)
        
        return image
    
    def draw_opening(self, image: np.ndarray, opening: Dict) -> np.ndarray:
        """
        Отрисовка отверстия (двери/окна) на изображении.
        
        Args:
            image: Изображение для отрисовки
            opening: Словарь с данными отверстия
        
        Returns:
            Изображение с нарисованным отверстием
        """
        points = opening.get("points", [])
        if len(points) < 2:
            return image
        
        # Определение цвета по типу
        opening_type = opening.get("type", "opening")
        color = self.opening_colors.get(opening_type, self.opening_color)
        
        # Конвертация точек
        pts = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
        
        # Отрисовка контура
        if len(pts) >= 3:
            # Замкнутый полигон
            cv2.polylines(image, [pts], True, color, self.opening_thickness)
            # Опционально: заливка с прозрачностью
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        else:
            # Линия
            cv2.line(image, tuple(pts[0]), tuple(pts[1]), 
                    color, self.opening_thickness)
        
        # Добавление метки
        if self.show_labels:
            opening_id = opening.get("id", "o?")
            opening_type = opening.get("type", "opening")
            label = f"{opening_id} ({opening_type})"
            
            # Позиция метки - центр bounding box или первая точка
            if "bbox" in opening:
                bbox = opening["bbox"]
                label_pos = (bbox["x"] + bbox["width"] // 2, 
                           bbox["y"] + bbox["height"] // 2)
            else:
                label_pos = (pts[0][0], pts[0][1])
            
            # Фон для текста для лучшей читаемости
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(image, 
                         (label_pos[0] - 2, label_pos[1] - text_height - 2),
                         (label_pos[0] + text_width + 2, label_pos[1] + 2),
                         (255, 255, 255), -1)
            cv2.putText(image, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image
    
    def visualize_result(self, 
                        image_path: str,
                        result: Dict,
                        output_path: Optional[str] = None,
                        show_openings: bool = True) -> np.ndarray:
        """
        Визуализация результата парсинга одного изображения.
        
        Args:
            image_path: Путь к исходному изображению
            result: Результат парсинга (словарь с "meta", "walls", "openings")
            output_path: Путь для сохранения результата (если None, не сохраняется)
            show_openings: Показывать ли отверстия
        
        Returns:
            Изображение с визуализацией
        """
        # Загрузка исходного изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Создание копии для отрисовки
        vis_image = image.copy()
        
        # Отрисовка стен
        walls = result.get("walls", [])
        print(f"Отрисовка {len(walls)} стен...")
        for wall in walls:
            vis_image = self.draw_wall(vis_image, wall)
        
        # Отрисовка отверстий
        if show_openings and "openings" in result:
            openings = result.get("openings", [])
            print(f"Отрисовка {len(openings)} отверстий...")
            for opening in openings:
                vis_image = self.draw_opening(vis_image, opening)
        
        # Добавление легенды
        vis_image = self._add_legend(vis_image, show_openings)
        
        # Сохранение результата
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Результат сохранен: {output_path}")
        
        return vis_image
    
    def _add_legend(self, image: np.ndarray, show_openings: bool) -> np.ndarray:
        """
        Добавление легенды на изображение.
        
        Args:
            image: Изображение
            show_openings: Показывать ли информацию об отверстиях (не используется, оставлено для совместимости)
        
        Returns:
            Изображение с легендой
        """
        h, w = image.shape[:2]
        
        # Параметры легенды
        legend_x = 10
        legend_y = 30
        line_height = 25
        font_scale = 0.5
        font_thickness = 1
        
        # Фон для легенды (только для стен)
        legend_height = 40
        cv2.rectangle(image, 
                     (legend_x - 5, legend_y - 20),
                     (legend_x + 200, legend_y + legend_height),
                     (255, 255, 255), -1)
        cv2.rectangle(image, 
                     (legend_x - 5, legend_y - 20),
                     (legend_x + 200, legend_y + legend_height),
                     (0, 0, 0), 2)
        
        # Заголовок
        cv2.putText(image, "Legend:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        # Стены
        y_pos = legend_y + line_height
        if self.fill_walls:
            # Показываем заполненный прямоугольник
            cv2.rectangle(image, (legend_x, y_pos - 8), 
                         (legend_x + 30, y_pos + 2), self.wall_color, -1)
            cv2.rectangle(image, (legend_x, y_pos - 8), 
                         (legend_x + 30, y_pos + 2), self.wall_color, self.wall_thickness)
        else:
            # Показываем линию
            cv2.line(image, (legend_x, y_pos), (legend_x + 30, y_pos),
                    self.wall_color, self.wall_thickness)
        cv2.putText(image, "Walls", (legend_x + 35, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        return image
    
    def visualize_from_json(self,
                           json_path: str,
                           images_dir: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           show_openings: bool = True) -> List[np.ndarray]:
        """
        Визуализация результатов из JSON файла.
        
        Args:
            json_path: Путь к JSON файлу с результатами
            images_dir: Директория с исходными изображениями (если None, используется текущая)
            output_dir: Директория для сохранения результатов (если None, не сохраняется)
            show_openings: Показывать ли отверстия
        
        Returns:
            Список визуализированных изображений
        """
        # Загрузка JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not isinstance(results, list):
            results = [results]
        
        # Создание выходной директории
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        visualized = []
        
        for result in results:
            source = result.get("meta", {}).get("source", "")
            if not source:
                print("Пропущен результат без указания источника")
                continue
            
            # Поиск пути к изображению
            if images_dir:
                image_path = os.path.join(images_dir, source)
            else:
                image_path = source
            
            if not os.path.exists(image_path):
                print(f"Изображение не найдено: {image_path}")
                continue
            
            # Определение пути для сохранения
            output_path = None
            if output_dir:
                base_name = os.path.splitext(source)[0]
                output_path = os.path.join(output_dir, f"{base_name}_visualized.png")
            
            # Визуализация
            try:
                vis_image = self.visualize_result(
                    image_path, result, output_path, show_openings)
                visualized.append(vis_image)
                print(f"✓ Визуализировано: {source}")
            except Exception as e:
                print(f"✗ Ошибка при визуализации {source}: {e}")
        
        return visualized


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация результатов парсинга планов этажей")
    parser.add_argument("--image", "-img", required=True,
                       help="Путь к исходному изображению")
    parser.add_argument("--json", "-j", required=True,
                       help="Путь к JSON файлу с результатами")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Путь к папке для сохранения результата")
    parser.add_argument("--no-openings", action="store_true",
                       help="Не показывать отверстия (двери/окна)")
    parser.add_argument("--no-labels", action="store_true",
                       help="Не показывать ID метки")
    parser.add_argument("--wall-color", nargs=3, type=int, default=[0, 255, 0],
                       help="Цвет стен в формате BGR (по умолчанию: 0 255 0 - зеленый)")
    parser.add_argument("--wall-thickness", type=int, default=2,
                       help="Толщина линий стен (по умолчанию: 2)")
    parser.add_argument("--no-fill-walls", action="store_true",
                       help="Не заполнять полигоны стен (только контуры)")
    parser.add_argument("--wall-opacity", type=float, default=0.5,
                       help="Прозрачность заливки стен (0.0-1.0, по умолчанию: 0.5)")
    
    args = parser.parse_args()
    
    # Проверка существования файлов
    if not os.path.exists(args.image):
        print(f"Ошибка: Изображение не найдено: {args.image}")
        return
    
    if not os.path.exists(args.json):
        print(f"Ошибка: JSON файл не найден: {args.json}")
        return
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загрузка JSON
    with open(args.json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Если JSON содержит список, берем первый результат
    # Если словарь - используем его напрямую
    if isinstance(results, list):
        if len(results) == 0:
            print("Ошибка: JSON файл не содержит результатов")
            return
        result = results[0]
    else:
        result = results
    
    # Формирование имени выходного файла: оригинальное_имя + "visualized" + расширение
    image_name = os.path.basename(args.image)
    base_name, ext = os.path.splitext(image_name)
    output_filename = f"{base_name}_visualized{ext}"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Создание визуализатора
    visualizer = FloorPlanVisualizer(
        wall_color=tuple(args.wall_color),
        wall_thickness=args.wall_thickness,
        show_labels=not args.no_labels,
        fill_walls=not args.no_fill_walls,
        wall_fill_opacity=args.wall_opacity
    )
    
    # Визуализация
    print(f"Визуализация изображения: {args.image}")
    try:
        vis_image = visualizer.visualize_result(
            args.image,
            result,
            output_path=output_path,
            show_openings=not args.no_openings
        )
        print(f"✓ Результат сохранен: {output_path}")
    except Exception as e:
        print(f"✗ Ошибка при визуализации: {e}")


if __name__ == "__main__":
    main()


