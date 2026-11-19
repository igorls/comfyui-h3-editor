"""Submit one reproducible, headless Flash Ref2VA character or pose benchmark."""

import argparse
import json

from run_take import MODELS, TAKE, call, graph

SWAP_LORA = "h3_character_swap_v1.5.safetensors"
DETECTOR = "rt_detr_v4-x-hgnet_fp16.safetensors"
POSE = "sdpose_wholebody_fp16.safetensors"
CONTROL = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
EDITOR_NODES = {
    "H3Take": "H3EditorTake",
    "H3HyperFlowLoader": "H3EditorHyperFlowLoader",
    "H3HyperFlowSigmas": "H3EditorHyperFlowSigmas",
}


def benchmark_graph(args):
    take = dict(TAKE, prompt=args.prompt, width=args.width, height=args.height, length=args.frames)
    recipe = argparse.Namespace(unet=MODELS["unet"], shift=12.0, steps=args.steps, sampler="euler",
                                seed=args.seed, prefix=args.prefix, source="", start="", mid=[], ref=[], region="")
    prompt = graph(recipe, take)
    for node in prompt.values():
        node["class_type"] = EDITOR_NODES.get(node["class_type"], node["class_type"])

    def add(node_id, class_type, **inputs):
        prompt[node_id] = {"class_type": class_type, "inputs": inputs}

    condition = prompt["10"]["inputs"]
    add("40", "H3EditorLoadStill", file=args.character)
    condition["ref_images.ref_image_0"] = ["40", 0]
    add("213", "H3EditorLoadClip", file=args.video, first=args.first, last=args.first + args.frames, reference_only=True)

    if args.mode == "pose":
        add("215", "UNETLoader", unet_name=DETECTOR, weight_dtype="default")
        add("216", "RTDETR_detect", model=["215", 0], image=["213", 0], threshold=0.5,
            class_name="person", max_detections=1)
        add("217", "CheckpointLoaderSimple", ckpt_name=POSE)
        add("218", "SDPoseKeypointExtractor", model=["217", 0], vae=["217", 2], image=["213", 0],
            bboxes=["216", 0], batch_size=8)
        add("219", "SDPoseDrawKeypoints", keypoints=["218", 0], draw_body=True, draw_head=True, draw_hands=True,
            draw_feet=True, draw_face=False, stick_width=4, face_point_size=2, score_threshold=0.3)
        add("220", "ModelPatchLoader", name=CONTROL)
        add("221", "H3FlashPoseBench", model=["5", 0], vae=["3", 0], model_patch=["220", 0],
            control_video=["219", 0], strength=args.strength)
        prompt["13"]["inputs"]["model"] = ["221", 0]
        add("222", "CreateVideo", images=["219", 0], fps=24.0)
        add("223", "SaveVideo", video=["222", 0], filename_prefix=args.prefix + "-guide", format="mp4", codec="h264")
    else:
        condition["ref_videos.ref_video_0"] = ["213", 0]
        if args.mode == "swap":
            add("230", "H3EditorLora", model=["1", 0], lora_name=SWAP_LORA, strength=1.0)
            prompt["5"]["inputs"]["model"] = ["230", 0]
    return prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--mode", choices=("reference", "swap", "pose"), required=True)
    parser.add_argument("--video", required=True, help="24 fps source path relative to ComfyUI/output")
    parser.add_argument("--character", required=True, help="still path relative to ComfyUI/output")
    parser.add_argument("--prefix", required=True, help="ComfyUI output filename prefix")
    parser.add_argument("--prompt", required=True, help="full H3 prompt, with <Picture 1> and optionally <Video 1>")
    parser.add_argument("--first", type=int, default=0, help="first source frame")
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--steps", type=int, choices=(4, 6, 8), default=6)
    parser.add_argument("--strength", type=float, default=1.0, help="pose-control strength")
    parser.add_argument("--seed", type=int, default=5241)
    parser.add_argument("--dry-run", action="store_true", help="print the API graph without queueing it")
    args = parser.parse_args()

    if args.first < 0 or not 48 <= args.frames <= 360 or (args.frames - 5) % 17:
        parser.error("choose a nonnegative first frame and a 2–15 second length on the 17n+5 frame grid")
    if any(side < 32 or side % 32 for side in (args.width, args.height)):
        parser.error("width and height must be positive multiples of 32")
    if not 0 <= args.strength <= 2:
        parser.error("strength must be between 0 and 2")

    prompt = benchmark_graph(args)
    result = prompt if args.dry_run else call(args.url.rstrip("/") + "/prompt", {"prompt": prompt})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
