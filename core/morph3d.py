import cv2
import dlib
import numpy as np
import tempfile
from tqdm import tqdm
import os

class FaceMorpher3D:
    def __init__(self, predictor_path="../models/shape_predictor_68_face_landmarks.dat"):
        self.predictor_path = os.path.abspath(predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        
        if not os.path.exists(self.predictor_path):
            raise FileNotFoundError(f"Файл предсказателя не найден: {self.predictor_path}")
        
        self.predictor = dlib.shape_predictor(self.predictor_path)
    
    def get_facial_landmarks(self, image):
        """Получение лицевых landmarks"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            # Попытка с Haar cascades как fallback
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces_haar = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces_haar) == 0:
                raise ValueError("Лицо не обнаружено на изображении")
            
            x, y, w, h = faces_haar[0]
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        else:
            rect = faces[0]
        
        landmarks = self.predictor(gray, rect)
        return np.array([(p.x, p.y) for p in landmarks.parts()])
    
    def create_animation(self, image_path, output_path=None, duration=4.0):
        """Создание 3D-анимации"""
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Получение landmarks
        try:
            landmarks = self.get_facial_landmarks(image)
        except Exception as e:
            raise ValueError(f"Ошибка детекции лица: {str(e)}")
        
        # Настройка выходного файла
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp4')
        
        # Параметры видео
        fps = 30
        total_frames = int(duration * fps)
        height, width = image.shape[:2]
        
        # Создание видео writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Генерация анимации
        for frame_idx in tqdm(range(total_frames), desc="Создание 3D анимации"):
            progress = frame_idx / total_frames
            
            # Параметры трансформации
            angle = progress * 360  # Полный оборот
            scale = 0.9 + 0.1 * np.sin(progress * 4 * np.pi)  # Пульсация
            
            # Центр лица (приблизительно)
            face_center = np.mean(landmarks, axis=0).astype(int)
            center = (int(face_center[0]), int(face_center[1]))
            
            # Матрица поворота и масштабирования
            M = cv2.getRotationMatrix2D(center, angle, scale)
            
            # Применение аффинного преобразования
            warped = cv2.warpAffine(image, M, (width, height))
            
            # Эффект моргания
            if 0.3 < (progress * 2) % 1.0 < 0.4:
                warped = self._apply_blink_effect(warped, landmarks, M)
            
            out.write(warped)
        
        out.release()
        return output_path
    
    def _apply_blink_effect(self, image, landmarks, transform_matrix):
        """Применение эффекта моргания"""
        # Преобразование landmarks
        ones = np.ones((landmarks.shape[0], 1))
        points_homogeneous = np.hstack([landmarks, ones])
        transformed_landmarks = np.dot(transform_matrix, points_homogeneous.T).T
        
        # Индексы глаз
        left_eye_indices = [36, 37, 38, 39, 40, 41]
        right_eye_indices = [42, 43, 44, 45, 46, 47]
        
        # Зарисовка глаз
        for eye_indices in [left_eye_indices, right_eye_indices]:
            eye_points = transformed_landmarks[eye_indices].astype(np.int32)
            if len(eye_points) >= 3:
                hull = cv2.convexHull(eye_points)
                cv2.fillPoly(image, [hull], (0, 0, 0))
        
        return image

# Функция для импорта
def create_3d_animation(image_path):
    morpher = FaceMorpher3D()
    return morpher.create_animation(image_path)
