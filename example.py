import os
# export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
if "FASTVIDEO_ATTENTION_BACKEND" not in os.environ:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

from fastvideo import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.configs.sample.teacache import WanTeaCacheParams

def main():

    SAVE_VIDEO=False
    OUTPUT_PATH="my_videos/"

    # resolution=(256,256)        # tiny
    resolution=(640,480)          # 480p
    # resolution=(1280,720)       # 720p
    WIDTH=resolution[0]
    HEIGHT=resolution[1]

    # NUM_FRAMES=250
    NUM_FRAMES=125               # default

    # NUM_STEPS=50               # default
    NUM_STEPS=4                  # distilled

    # TeaCache settings
    USE_TEACACHE=False
    TEACACHE_THRESHOLD=0.08
    new_teacache_params = WanTeaCacheParams(teacache_thresh=TEACACHE_THRESHOLD)

    print('')
    print('FASTVIDEO_ATTENTION_BACKEND:', os.environ["FASTVIDEO_ATTENTION_BACKEND"])
    print('SAVE_VIDEO:', SAVE_VIDEO)
    print('resolution:', resolution)
    print('frames:', NUM_FRAMES)
    # print('steps:', NUM_STEPS)
    print('TeaCache:', USE_TEACACHE, TEACACHE_THRESHOLD)
    print('Sequence Parallelism (sp_size): 1 (disabled)')
    print('')

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
        sp_size=1,  # Disable sequence parallelism (set to 1)
        use_fsdp_inference=False,  # Disable FSDP inference
        VSA_sparsity=0.90,
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        output_path=OUTPUT_PATH,
        height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES,
        # num_inference_steps=NUM_STEPS,
        save_video=SAVE_VIDEO,
        output_type="latent",
        teacache_params=new_teacache_params,
        enable_teacache=USE_TEACACHE
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
        # num_inference_steps=NUM_STEPS,
        save_video=SAVE_VIDEO,
        output_type="latent",
        teacache_params=new_teacache_params,
        enable_teacache=USE_TEACACHE
    )

if __name__ == '__main__':
    main()
