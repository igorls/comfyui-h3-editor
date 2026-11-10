"""Queue one H3 Take on a running ComfyUI and wait for the result.

python tools/run_take.py --prefix h3take/base --prompt "..." --set length=124
python tools/run_take.py --prefix h3take/ext --source base.mp4 --set video_mode=extend_after --set length=192

--source names a file in ComfyUI's input folder and connects its frames and audio.

The take renders on HyperFlow H3 Flash INT8, chosen as the editor chooses: the fl2va file for a
prompt alone or a source to carry on from, the ref2va file once a picture is connected. --unet
names another file. One that is not a HyperFlow checkpoint loads with ComfyUI's own loader and
the simple scheduler.
"""

import argparse
import json
import time
import urllib.request

MODELS = {
    "unet": "hyperflow/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
    "unet_text": "hyperflow/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}

TAKE = {
    "prompt": "",
    "width": 1344,
    "height": 768,
    "length": 124,
    "checkpoint": "ref2va",
    "anchor": "none",
    "mid_frames": "",
    "video_mode": "off",
    "hold_frames": 39,
    "strength": "hold",
    "keep": 0.0,
    "align": "av",
    "video_feather": 0,
    "audio_hold": "off",
    "audio_feather": 8,
    "ref_image_size": "match",
}


def file_name(name):
    return name.replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_hyperflow(name):
    """Whether the pack's HyperFlow loader reads this file: any .gguf, or a file with hyperflow in its name."""
    return file_name(name).endswith(".gguf") or "hyperflow" in file_name(name)


def graph(args, take):
    if is_hyperflow(args.unet):
        loader = {"class_type": "H3HyperFlowLoader", "inputs": {"unet_name": args.unet, "weights_in": "auto"}}
        sigmas = {"class_type": "H3HyperFlowSigmas", "inputs": {"model": ["5", 0], "steps": args.steps}}
    else:
        loader = {"class_type": "UNETLoader", "inputs": {"unet_name": args.unet, "weight_dtype": "default"}}
        sigmas = {"class_type": "BasicScheduler", "inputs": {
            "model": ["5", 0], "scheduler": "simple", "steps": args.steps, "denoise": 1.0}}
    g = {
        "1": loader,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": args.shift, "shift_audio": 3.0}},
        "10": {"class_type": "H3Take", "inputs": dict(take, clip=["2", 0], vae=["3", 0], audio_vae=["4", 0])},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}},
        "12": sigmas,
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": ["10", 0]}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": 24.0, "audio": ["17", 0]}},
        "19": {"class_type": "SaveVideo", "inputs": {"video": ["18", 0], "filename_prefix": args.prefix, "format": "auto", "codec": "auto"}},
    }
    if args.source:
        g["6"] = {"class_type": "LoadVideo", "inputs": {"file": args.source}}
        g["7"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["6", 0]}}
        g["10"]["inputs"]["source_video"] = ["7", 0]
        g["10"]["inputs"]["source_audio"] = ["7", 1]
    # Autogrow sockets are addressed as group.name in API format.
    images = [("start_image", args.start)] if args.start else []
    images += [("mid_images.mid_image_%s" % i, name) for i, name in enumerate(args.mid)]
    images += [("ref_images.ref_image_%s" % i, name) for i, name in enumerate(args.ref)]
    for i, (socket, name) in enumerate(images):
        node = str(30 + i)
        g[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["10"]["inputs"][socket] = [node, 0]
    if args.region:
        width, height = take["width"], take["height"]
        g["20"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": width, "height": height}}
        g["21"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": width // 2, "height": height}}
        g["22"] = {"class_type": "MaskComposite", "inputs": {
            "destination": ["20", 0], "source": ["21", 0], "operation": "add",
            "x": 0 if args.region == "left" else width // 2, "y": 0}}
        g["10"]["inputs"]["region"] = ["22", 0]
    return g


def call(url, data=None):
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit("HTTP %s: %s" % (error.code, error.read().decode()[:1500]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--start", default="", help="input file for start_image")
    parser.add_argument("--mid", action="append", default=[], help="input file for the next mid image")
    parser.add_argument("--ref", action="append", default=[], help="input file for the next reference image")
    parser.add_argument("--region", choices=["left", "right"], default="", help="regenerate this half of the held video")
    parser.add_argument("--unet", default="", help="diffusion model file; the HyperFlow INT8 file that fits the take when empty")
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--steps", type=int, default=0, help="6 on HyperFlow (4, 6 or 8), 8 on other checkpoints when 0")
    parser.add_argument("--shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--set", action="append", default=[], help="H3Take input, name=value")
    args = parser.parse_args()

    take = dict(TAKE, prompt=args.prompt)
    for item in args.set:
        name, value = item.split("=", 1)
        take[name] = type(TAKE[name])(value)
    if not any(item.startswith("checkpoint=") for item in args.set):
        if args.unet:
            take["checkpoint"] = "fl2va" if "fl2va" in file_name(args.unet) else "ref2va"
        else:
            take["checkpoint"] = "ref2va" if args.start or args.mid or args.ref else "fl2va"
    args.unet = args.unet or MODELS["unet" if take["checkpoint"] == "ref2va" else "unet_text"]
    args.steps = args.steps or (6 if is_hyperflow(args.unet) else 8)

    started = time.time()
    prompt_id = call(args.url + "/prompt", {"prompt": graph(args, take)})["prompt_id"]
    while time.time() - started < args.timeout:
        history = call("%s/history/%s" % (args.url, prompt_id)).get(prompt_id)
        if history:
            status = history.get("status", {})
            files = []
            labels = ""
            for output in history.get("outputs", {}).values():
                for group in output.values():
                    for item in group if isinstance(group, list) else []:
                        if isinstance(item, dict) and "filename" in item:
                            files.append("%s/%s" % (item.get("subfolder", ""), item["filename"]))
            errors = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
            print(json.dumps({
                "status": status.get("status_str"),
                "seconds": round(time.time() - started, 1),
                "files": files,
                "error": [{k: e.get(k) for k in ("node_type", "exception_type", "exception_message")} for e in errors],
            }, indent=1))
            return 1 if errors else 0
        time.sleep(3)
    raise SystemExit("timed out")


if __name__ == "__main__":
    main()
