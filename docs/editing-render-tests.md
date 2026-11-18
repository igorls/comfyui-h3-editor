# Testing interior frame guides and VOSR2 on H3 footage

Experiments run November 18, 2026 on the H3 Editor 0.12.0 base. Source footage is in [the realism showcase](media/realism). These experiments use generated footage distributed with the editor.

## Interior frame guides

H3 Take sent still images to MiniMaxH3AddGuide at the requested frame indices. All comparisons used HyperFlow H3 Flash INT8, Euler, six steps, shifts 12/3, 90 frames at 24 fps and a start guide. Within a pair, the starting picture, prompt, dimensions and seed stayed the same. The GPU was an RTX PRO 6000 Blackwell Workstation Edition running the project's Docker ComfyUI. For the golden-hour shot, the prompt described the subject walking and turning toward the camera. For the neon shot, it described her looking up in the rain. Both sets used guides extracted from the source shot, with FFmpeg scaling each still directly to the requested render dimensions.

Mean absolute error measures the difference from the guide at the requested frame, across all pixels and RGB channels on a 0–255 scale. Smaller means closer. The best match is the frame with the smallest error across the entire take.

| Clip and checkpoint | Size and seed | Guide frame | Start only | With interior guides | Best match |
| --- | --- | ---: | ---: | ---: | ---: |
| Golden hour, fl2va | 640 × 352, 61118 | 42 | 20.76 | 5.33 | 42 |
| Golden hour, fl2va | 640 × 352, 61119 | 42 | 33.63 | 5.50 | 42 |
| Neon rain, ref2va | 640 × 352, 61120 | 68 | 36.87 | 5.55 | 69 |
| Neon rain, ref2va, two guides | 640 × 352, 61120 | 34 | 24.22 | 5.36 | 34 |
| Neon rain, ref2va, two guides | 640 × 352, 61120 | 68 | 36.87 | 5.44 | 69 |
| Neon rain, ref2va, two guides | 1344 × 768, 61121 | 34 | 16.04 | 4.07 | 34 |
| Neon rain, ref2va, two guides | 1344 × 768, 61121 | 68 | 26.37 | 4.12 | 68 |

The earlier nine-frame freeze did not recur in these draft runs. In the golden-hour shots the median frame-to-frame change around the guide was 6.55 and 6.18 levels. The neon two-guide shot moved less throughout the take, including away from the guides. Around frames 34 and 68 its median changes were 0.95 and 1.33, compared with 1.08 away from either guide. A guide can influence the motion leading up to it and the amount of motion in the rest of the shot.

At H3's native large size, the two-guide take reached both targets on their exact frames. Around each guide the median adjacent-frame changes were 1.38 and 1.51, against 1.34 away from the guides. This is nine renders in total on two short shots. Hard changes of pose or scene and other checkpoints still need their own comparison.

Reproduce the generations with `tools/run_take.py`, supplying the same `--start`, checkpoint, seed and prompt to each pair. Add `--mid <image>` for each interior still, in order, and `--set mid_frames=34,68` for the two-guide condition. Use `--set anchor=start` for both the control and the guided shot.

## VOSR2

A dedicated Colab G4 with an RTX PRO 6000 Blackwell Server Edition ran [the VOSR2 ComfyUI implementation](https://github.com/ylchen333/ComfyUI-VOSR2/tree/6a21810b3f4f1a0ffc477c55330db0ac03c671f0) against ComfyUI commit `71096d1` using its model loader and inference function. Each source was downsampled to 640 × 360 with an area filter. The first 48 frames (two seconds) were upscaled to 1280 × 720 with BF16, 2× scale, wavelet color alignment, a 512-pixel DiT tile, a 1024-pixel VAE tile, and 32-pixel overlaps. A smaller 320 × 180 to 640 × 360 smoke test also completed.

The ordinary batch starts at seed 42 and increments the seed for each frame. The fixed-seed run calls the same inference function on one frame at a time, always using 42. Timings cover inference and transfer of the output tensor, excluding initial model downloads, input decoding and output encoding. Peak memory is PyTorch's peak allocated CUDA memory.

| Shot | Method | Seconds for 48 frames | Peak GiB | PSNR (dB) | Temporal error |
| --- | --- | ---: | ---: | ---: | ---: |
| Golden hour | Bicubic reference | — | — | 44.52 | 0.99 |
| Golden hour | VOSR2 batch | 18.28 | 5.05 | 37.48 | 2.11 |
| Golden hour | VOSR2 fixed seed | 17.65 | 4.56 | 37.45 | 2.07 |
| Neon rain | Bicubic reference | — | — | 41.36 | 0.75 |
| Neon rain | VOSR2 batch | 17.53 | 5.05 | 35.59 | 1.29 |
| Neon rain | VOSR2 fixed seed | 17.61 | 4.56 | 35.57 | 1.10 |

PSNR compares each output frame to the published source resized to the target dimensions. Temporal error is the mean absolute difference between consecutive output-frame changes and the corresponding source-frame changes, in 8-bit RGB levels. These measures reward fidelity to the source. They will also penalize invented detail, even when a viewer likes its appearance. In a separate eight-frame batch of one repeated rain frame, changing only the seed produced an average adjacent-frame difference of 0.74 levels.

On the neon close-up VOSR2 sharpens the eyes and hair and redraws skin and water texture. Keeping the seed fixed reduced temporal error in both examples. A video upscaler comparison on longer, naturally low-resolution footage would be the next test before offering VOSR2 as a built-in editor option. Neither the small benchmark nor the 720p run evaluated a final 1080p delivery or an editor integration.
