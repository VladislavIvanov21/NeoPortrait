import gradio as gr
import os
import tempfile
from core.animator import animate_portrait
from core.morph3d import create_3d_animation
from core.utils import validate_image, validate_video, create_thumbnail

# CSS для красивого интерфейса
css = """
footer {visibility: hidden}
.gradio-container {max-width: 1200px !important}
.output-video {
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.upload-box {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
"""

def process_animation(input_image, driving_video, mode, progress=gr.Progress()):
    """Обработка анимации с прогресс-баром"""
    try:
        # Валидация входных данных
        if not validate_image(input_image):
            raise ValueError("Некорректное изображение")
        
        if mode == "GAN" and not validate_video(driving_video):
            raise ValueError("Некорректное видео для GAN режима")
        
        progress(0.2, desc="Загрузка моделей...")
        
        # Выбор режима обработки
        if mode == "GAN":
            progress(0.4, desc="Создание GAN анимации...")
            output_path = animate_portrait(input_image, driving_video)
        else:
            progress(0.4, desc="Создание 3D морфинга...")
            output_path = create_3d_animation(input_image)
        
        progress(0.8, desc="Генерация превью...")
        
        # Создание превью
        thumbnail_path = tempfile.mktemp(suffix='.jpg')
        create_thumbnail(output_path, thumbnail_path)
        
        progress(1.0, desc="Готово!")
        return output_path, thumbnail_path
        
    except Exception as e:
        raise gr.Error(f"Ошибка обработки: {str(e)}")

# Создание интерфейса
with gr.Blocks(css=css, title="PortraitLive - Оживление портретов") as app:
    gr.Markdown("""
    # 🎭 PortraitLive
    ## Оживление портретов с помощью AI (GAN + 3D Morphing)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Входные данные")
            
            input_image = gr.Image(
                label="Исходный портрет",
                type="filepath",
                elem_classes="upload-box"
            )
            
            driving_video = gr.Video(
                label="Эталонное видео (только для GAN)",
                elem_classes="upload-box"
            )
            
            mode = gr.Radio(
                choices=["GAN", "3D Morph"],
                label="Режим анимации",
                value="GAN",
                interactive=True
            )
            
            process_btn = gr.Button(
                "🚀 Создать анимацию",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Результат")
            
            with gr.Row():
                output_thumbnail = gr.Image(
                    label="Превью",
                    interactive=False,
                    height=200
                )
                output_video = gr.Video(
                    label="Анимация",
                    elem_classes="output-video",
                    interactive=False
                )
            
            gr.Markdown("### 📋 Информация")
            info_text = gr.Textbox(
                label="Статус",
                value="Загрузите изображение и выберите режим",
                interactive=False
            )
    
    # Примеры
    examples = gr.Examples(
        examples=[
            [os.path.join("examples", "mona_lisa.jpg"), None, "3D Morph"],
            [os.path.join("examples", "portrait.jpg"), os.path.join("examples", "driving_video.mp4"), "GAN"]
        ],
        inputs=[input_image, driving_video, mode],
        outputs=[output_video, output_thumbnail],
        fn=process_animation,
        cache_examples=True
    )
    
    # Обработчики событий
    process_btn.click(
        fn=process_animation,
        inputs=[input_image, driving_video, mode],
        outputs=[output_video, output_thumbnail]
    )
    
    # Обновление информации
    input_image.change(
        fn=lambda x: f"Изображение загружено: {os.path.basename(x) if x else 'Нет'}", 
        inputs=input_image, 
        outputs=info_text
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
