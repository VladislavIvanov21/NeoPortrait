import cv2
import numpy as np
import os
from PIL import Image
import io
import base64

def validate_image(image_path):
    """Проверка валидности изображения"""
    if not os.path.exists(image_path):
        return False
    
    try:
        img = cv2.imread(image_path)
        return img is not None and img.size > 0
    except:
        return False

def validate_video(video_path):
    """Проверка валидности видео"""
    if not os.path.exists(video_path):
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Проверка хотя бы одного кадра
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except:
        return False

def create_thumbnail(video_path, output_path, time_sec=1):
    """Создание превью для видео"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        frame_pos = int(fps * time_sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            cv2.imwrite(output_path, frame)
            return True
        
        return False
    except:
        return False

def resize_image(image, max_size=512):
    """Изменение размера изображения с сохранением пропорций"""
    if image is None:
        return None
    
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def image_to_base64(image):
    """Конвертация изображения в base64"""
    try:
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            return base64.b64encode(encoded_image).decode('utf-8')
        return ""
    except:
        return ""

def cleanup_temp_files(file_paths):
    """Очистка временных файлов"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

def get_file_size_mb(file_path):
    """Получение размера файла в МБ"""
    if os.path.exists(file_path):
        return round(os.path.getsize(file_path) / (1024 * 1024), 2)
    return 0
