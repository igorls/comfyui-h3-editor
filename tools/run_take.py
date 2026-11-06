"""Queue one H3 Take on a running ComfyUI and wait for the result.

python tools/run_take.py --prefix h3take/base --prompt "..." --set length=124
python tools/run_take.py --prefix h3take/ext --source base.mp4 --set video_mode=extend_after --set length=192

--source names a file in ComfyUI's input folder and connects its frames and audio.
"""

import argparse
import json
import time
import urllib.request

MODELS = {
    "unet": "fasth3_omniref_qad_nvfp4_comfy.safetensors",
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
    "keep": 0.0,
    "align": "av",
    "video_feather": 0,
    "audio_hold": "off",
    "audio_feather": 8,
    "ref_image_size": "match",
}


def graph(args, take):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": args.unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": args.shift, "shift_audio": 3.0}},
        "10": {"class_type": "H3Take", "inputs": dict(take, clip=["2", 0], vae=["3", 0], audio_vae=["4", 0])},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": "simple", "steps": args.steps, "denoise": 1.0}},
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
    parser.add_argument("--unet", default=MODELS["unet"])
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--set", action="append", default=[], help="H3Take input, name=value")
    args = parser.parse_args()

    take = dict(TAKE, prompt=args.prompt)
    for item in args.set:
        name, value = item.split("=", 1)
        take[name] = type(TAKE[name])(value)

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
