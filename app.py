import gradio as gr
from core.animator import animate_portrait
from core.morph3d import morph_3d

def process_image(source_img, driving_video, mode):
    if mode == "GAN":
        return animate_portrait(source_img, driving_video)
    else:
        return morph_3d(source_img)

with gr.Blocks(title="PortraitLive", theme="soft") as app:
    gr.Markdown("# 🎭 PortraitLive: Оживление портретов")
    
    with gr.Row():
        with gr.Column():
            source = gr.Image(label="Исходный портрет", type="filepath")
            driving = gr.Video(label="Эталонное видео (для GAN)")
            mode = gr.Radio(["GAN", "3D Morph"], label="Режим")
            btn = gr.Button("Анимировать")
        
        with gr.Column():
            output = gr.Video(label="Результат")
    
    btn.click(
        process_image,
        inputs=[source, driving, mode],
        outputs=output
    )

app.launch(server_port=7860)
