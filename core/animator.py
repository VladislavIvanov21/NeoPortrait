import torch
from demo import load_checkpoints, make_animation
from skimage import img_as_ubyte

def animate_portrait(source_path, driving_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generator, kp_detector = load_checkpoints(
        config_path="config/vox-256.yaml",
        checkpoint_path="models/vox-cpk.pth.tar",
        device=device
    )
    
    source = imageio.imread(source_path)
    driving = imageio.mimread(driving_path)
    
    predictions = make_animation(
        source, driving, generator, kp_detector, device=device
    )
    
    output_path = "result.mp4"
    imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions])
    
    return output_path
