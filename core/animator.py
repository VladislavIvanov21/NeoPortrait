import torch
import imageio
import numpy as np
from skimage import img_as_ubyte
import tempfile
import os
import cv2
from tqdm import tqdm

class GANAnimator:
    def __init__(self, model_path="../models/vox-cpk.pth.tar"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.abspath(model_path)
        self.generator = None
        self.kp_detector = None
        self.is_loaded = False
        
    def _load_models(self):
        """Загрузка моделей с обработкой ошибок"""
        if self.is_loaded:
            return
            
        try:
            # Динамический импорт для избежания зависимостей
            from demo import load_checkpoints
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            
            print(f"Загрузка модели из {self.model_path}...")
            self.generator, self.kp_detector = load_checkpoints(
                config_path="config/vox-256.yaml",
                checkpoint_path=self.model_path,
                device=self.device
            )
            self.is_loaded = True
            print("Модели успешно загружены!")
            
        except ImportError:
            raise ImportError("Требуется установка дополнительных зависимостей: pip install git+https://github.com/AliaksandrSiarohin/first-order-model")
        except Exception as e:
            raise Exception(f"Ошибка загрузки моделей: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Предобработка изображения"""
        image = imageio.imread(image_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Конвертация в RGB если нужно
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
            
        return image
    
    def create_animation(self, source_path, driving_path, output_path=None):
        """Создание анимации"""
        self._load_models()
        
        # Загрузка данных
        source_image = self.preprocess_image(source_path)
        driving_video = imageio.mimread(driving_path, memtest=False)
        
        if len(driving_video) == 0:
            raise ValueError("Видео не содержит кадров")
        
        # Создание анимации
        from demo import make_animation
        predictions = make_animation(
            source_image,
            driving_video,
            self.generator,
            self.kp_detector,
            relative=True,
            adapt_movement_scale=True,
            device=self.device
        )
        
        # Сохранение результата
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp4')
        
        # Сохранение с прогресс-баром
        with imageio.get_writer(output_path, fps=30, quality=9) as writer:
            for i, frame in enumerate(tqdm(predictions, desc="Сохранение анимации")):
                writer.append_data(img_as_ubyte(frame))
        
        return output_path

# Функция для импорта
def animate_portrait(source_image, driving_video):
    animator = GANAnimator()
    return animator.create_animation(source_image, driving_video)
