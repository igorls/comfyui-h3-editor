# Flash motion transfer benchmark

This is a standalone ComfyUI node for reproducing the local Flash Ref2VA experiment. The H3 Editor uses its existing standard H3 motion-transfer path. Run this benchmark on a test instance of ComfyUI.

## Setup

1. Install H3 Editor and copy this directory as `ComfyUI/custom_nodes/h3-flash-pose-bench`. Start the test instance.
2. Use Flash Ref2VA INT8 and the text encoder and VAEs listed in [run_take.py](../run_take.py). The pose node reads only `adaln_t_table` from `minimax_h3_ref2va_pruned_int8_convrot.safetensors` in `models/diffusion_models`. Use the standard Ref2VA checkpoint with that table.
3. For pose mode, install `rt_detr_v4-x-hgnet_fp16.safetensors` in `models/diffusion_models`, `sdpose_wholebody_fp16.safetensors` in `models/checkpoints`, and `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` in `models/model_patches`. For swap mode, install `h3_character_swap_v1.5.safetensors` in `models/loras`. [ComfyUI pose workflow and downloads](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_fun_controlnet_union.json) · [Character Swap V1.5 model card](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
4. Put a 24 fps source video and a character still under the test instance’s output folder. The example uses the paths `bench/source.mp4` and `bench/character.png` within that folder.

## Run matched comparisons

Use the same source range, character still, seed, output size, and step count. For an A/B test of the LoRA, run `reference` and `swap` with identical prompts. For `pose`, describe the shot and character without referring to `<Video 1>`. That mode sends the source through pose extraction. It saves the rendered take and the skeleton video.

```powershell
$source = @('--video', 'bench/source.mp4', '--character', 'bench/character.png')
$prompt = 'Replace the performer in <Video 1> with the anime character in <Picture 1>. Keep the camera and performance.'
python tools/motion_transfer_bench.py @source --mode reference --prefix bench/reference --prompt $prompt
python tools/motion_transfer_bench.py @source --mode swap --prefix bench/swap --prompt $prompt
python tools/motion_transfer_bench.py @source --mode pose --prefix bench/pose --prompt 'The anime character from <Picture 1> performs the tracked choreography in a pale studio.'
```

The helper prints a ComfyUI prompt ID. Check `/history/{prompt_id}` for its result. Use `--dry-run` to save the graph before submitting it. The defaults match the portrait benchmark at 384 × 672, 124 frames, six steps, seed 5241, pose strength 1.0, and Euler sampling. The MP4s are raw samples with generated audio. Use the source audio for visual comparisons.

## Compatibility and evidence

The pose adapter feeds the standard H3 eight-channel timestep curve to the ControlNet while Flash uses its own 16-channel embedding. It remaps the five ControlNet injection points to Flash Ref2VA blocks 0, 9, 19, 29, and 39. The node targets this 49-block Flash variant and the five-block ControlNet v1.

The generated dance rendered at six and eight steps. A four-step run stalled and was interrupted. Both supplied portrait excerpts rendered in all three benchmark modes at six steps. The studio pose render borrowed the beach from the character still. The rooftop pose render changed the camera and duplicated the character briefly. The two RGB video-reference variants kept the rooftop turn, pan, and subject exit, with more photographic shading on the character. These are observations from single-seed examples. They leave open whether tracking, conditioning, or generation is the primary cause of each failure.
