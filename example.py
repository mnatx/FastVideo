import os
from fastvideo import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs

def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
        use_fsdp_inference=False,  # Disable FSDP inference
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()
