import os
from fastvideo import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs

def main():

    # export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
    # export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
    # export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

    if "FASTVIDEO_ATTENTION_BACKEND" not in os.environ:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

    SAVE_VIDEO=False
    OUTPUT_PATH="my_videos/"

    resolution=(256,256)        # tiny
    # resolution=(640,480)        # 480p
    # resolution=(1280,720)       # 720p
    # resolution=(1920,1080)      # 1080p
    WIDTH=resolution[0]
    HEIGHT=resolution[1]
    NUM_FRAMES=125                # default
    # NUM_FRAMES=250

    print('')
    print('FASTVIDEO_ATTENTION_BACKEND:', os.environ["FASTVIDEO_ATTENTION_BACKEND"])
    print('SAVE_VIDEO:', SAVE_VIDEO)
    print('resolution:', resolution)
    print('frames:', NUM_FRAMES)
    print('')

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
        use_fsdp_inference=False,  # Disable FSDP inference
        VSA_sparsity=0.90,
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        output_path=OUTPUT_PATH,  # Controls where videos are saved
        height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES,
        save_video=SAVE_VIDEO,
        output_type="latent"
    )

    # Generate another video with a different prompt, without reloading the model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")

    video2 = generator.generate_video(
        prompt2, 
        output_path=OUTPUT_PATH,
        height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES,
        save_video=SAVE_VIDEO,
        output_type="latent"
    )

if __name__ == '__main__':
    main()
