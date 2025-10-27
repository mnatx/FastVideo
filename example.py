import os
import torch
from fastvideo import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs

def detect_rocm_device_type():
    """Detect the specific ROCm device type for optimal configuration."""
    if not torch.cuda.is_available():
        return "unknown"
    
    device_name = torch.cuda.get_device_name().lower()
    
    # Check for specific device patterns (order matters for overlapping names)
    if "mi300x" in device_name:
        return "mi300x"
    elif "mi300" in device_name:
        return "mi300x"  # Default MI300 to MI300X
    elif "mi250" in device_name or "m250" in device_name:
        return "mi250"
    elif "mi210" in device_name:
        return "mi210"
    elif "w7800" in device_name or "radeon pro w7800" in device_name:
        return "w7800"
    elif "radeon pro" in device_name:
        return "w7800"
    else:
        return "generic_rocm"

def get_device_optimal_params(device_type):
    """Get optimal parameters for the detected device type."""
    params = {
        "mi300x": {
            "height": 512,
            "width": 512,
            "video_length": 32,
            "infer_steps": 4,
            "guidance_scale": 7.5,
            "description": "MI300X: Ultra-high performance with 192GB memory"
        },
        "mi250": {
            "height": 512,
            "width": 512,
            "video_length": 24,
            "infer_steps": 4,
            "guidance_scale": 7.5,
            "description": "MI250: High performance with 128GB memory and 128KB shared memory"
        },
        "mi210": {
            "height": 256,
            "width": 256,
            "video_length": 16,
            "infer_steps": 1,
            "guidance_scale": 2.0,
            "description": "MI210: Balanced performance with 64GB memory and 64KB shared memory"
        },
        "w7800": {
            "height": 256,
            "width": 256,
            "video_length": 8,
            "infer_steps": 1,
            "guidance_scale": 2.0,
            "description": "W7800: Conservative configuration with 30GB memory and 32KB shared memory"
        },
        "generic_rocm": {
            "height": 256,
            "width": 256,
            "video_length": 16,
            "infer_steps": 1,
            "guidance_scale": 2.0,
            "description": "Generic ROCm: Conservative configuration for compatibility"
        }
    }
    return params.get(device_type, params["generic_rocm"])

def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

    # Detect device type and get optimal parameters
    device_type = detect_rocm_device_type()
    params = get_device_optimal_params(device_type)
    
    print(f"Detected device: {device_type.upper()}")
    print(f"Using: {params['description']}")
    print(f"Parameters: {params['height']}x{params['width']}, {params['video_length']} frames, {params['infer_steps']} steps")

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
        use_fsdp_inference=False,  # Disable FSDP inference
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video with device-specific parameters
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True,
        height=params['height'],
        width=params['width'],
        video_length=params['video_length'],
        infer_steps=params['infer_steps'],
        guidance_scale=params['guidance_scale']
    )

if __name__ == '__main__':
    main()
