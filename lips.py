"""Whether a take's lips follow its sound, measured with SyncNet. The same file always gives the same numbers.

SyncNet (Chung and Zisserman, "Out of time: automated lip sync in the wild", 2016) embeds five frames of a face and
the 0.2 seconds of sound around them, and puts the two close together when the mouth makes that sound. Sliding the
sound against the picture finds the offset where they match best, and how much better that offset is than the rest
says how sure the match is. This is the measure lip-sync research reports as LSE-D and LSE-C.

  offset  frames (at 24 fps) between where the lips are and where the sound is: positive when the lips come after the
          sound, negative when they come before it. Within a frame or two is in sync.
  conf    how clearly the lips match the sound at that offset: the median distance over all offsets less the best.
          Real talking video reads about 6 to 9; H3 takes of a singer that look right read about 2 to 5, and ones
          whose lips do not follow the song under 1.

The network is written out here from joonson/syncnet_python (MIT, Copyright (c) 2016-present Joon Son Chung). Its
weights are not part of the pack: the Visual Geometry Group offers them for research under CC BY 4.0. Put
syncnet_v2.model in ComfyUI's models/syncnet folder, beside OpenCV's YuNet face detector,
face_detection_yunet_2023mar.onnx (MIT). Without them nothing is measured.

Usage: python lips.py take.mp4 [first last]     prints offset and confidence for frames first..last
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

FPS = 24            # the editor's frame rate
SYNC_FPS = 25       # what SyncNet was trained on: the frames are resampled to it
RATE = 16000
VSHIFT = 15         # SyncNet frames the sound is slid each way
MODEL_FILE = "syncnet_v2.model"
FACE_FILE = "face_detection_yunet_2023mar.onnx"
CROP = 3.6          # the square around the face, in eye-to-mouth distances
LOWER = 0.5         # how far below midway between eyes and mouth its centre sits, in the same distances
MIN_FRAMES = 24     # under a second has too little to judge
# How sure a match must be, and how near, to call the lips in sync; under SURE there is no match to speak of.
SURE = 1.5
CLOSE = 2


def plain(args):
    return args


def folder():
    """Where the two files live: ComfyUI's models/syncnet, or H3_SYNCNET_DIR."""
    if os.environ.get("H3_SYNCNET_DIR"):
        return os.environ["H3_SYNCNET_DIR"]
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "syncnet")
    except ImportError:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "syncnet")


def missing():
    """The files the measure needs and does not have, by name."""
    return [name for name in (MODEL_FILE, FACE_FILE) if not os.path.exists(os.path.join(folder(), name))]


# --- Sound: MFCCs as python_speech_features.mfcc makes them by default, which is what SyncNet was trained on. ---

def _mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def _hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mfcc(signal, rate=RATE, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, preemph=0.97, lifter=22):
    """(frames, 13) cepstra of a signal at int16 scale, ten milliseconds apart, the first replaced by log energy."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = np.append(signal[:1], signal[1:] - preemph * signal[:-1])
    size, step = int(round(winlen * rate)), int(round(winstep * rate))
    count = 1 if len(signal) <= size else 1 + int(math.ceil((len(signal) - size) / float(step)))
    padded = np.concatenate([signal, np.zeros((count - 1) * step + size - len(signal))])
    index = np.arange(size)[None, :] + step * np.arange(count)[:, None]
    frames = padded[index]
    power = (np.abs(np.fft.rfft(frames, nfft)) ** 2) / nfft
    energy = power.sum(1)
    energy[energy == 0] = np.finfo(float).eps
    points = _hz(np.linspace(_mel(0), _mel(rate / 2.0), nfilt + 2))
    bins = np.floor((nfft + 1) * points / rate).astype(int)
    bank = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        for i in range(bins[j], bins[j + 1]):
            bank[j, i] = (i - bins[j]) / float(bins[j + 1] - bins[j])
        for i in range(bins[j + 1], bins[j + 2]):
            bank[j, i] = (bins[j + 2] - i) / float(bins[j + 2] - bins[j + 1])
    feat = power.dot(bank.T)
    feat[feat == 0] = np.finfo(float).eps
    feat = np.log(feat)
    # DCT-II, orthonormal, as scipy.fftpack.dct(type=2, norm="ortho").
    n = feat.shape[1]
    k = np.arange(numcep)[:, None]
    basis = np.cos(np.pi * k * (2 * np.arange(n)[None, :] + 1) / (2.0 * n)) * np.sqrt(2.0 / n)
    basis[0] /= np.sqrt(2.0)
    cep = feat.dot(basis.T)
    cep *= 1 + (lifter / 2.0) * np.sin(np.pi * np.arange(numcep) / lifter)
    cep[:, 0] = np.log(energy)
    return cep


# --- The network ---

_NET = {}


def network():
    """SyncNet with its weights, loaded once, on the CPU: it is small, and the GPU belongs to the renders."""
    if "net" in _NET:
        return _NET["net"]
    import torch
    from torch import nn

    def block2(i, o, k, s=(1, 1), p=(1, 1)):
        return [nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

    def block3(i, o, k, s=(1, 1, 1), p=0):
        return [nn.Conv3d(i, o, kernel_size=k, stride=s, padding=p), nn.BatchNorm3d(o), nn.ReLU(inplace=True)]

    class S(nn.Module):
        def __init__(self):
            super().__init__()
            self.netcnnaud = nn.Sequential(
                *block2(1, 64, (3, 3)), nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
                *block2(64, 192, (3, 3)), nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
                *block2(192, 384, (3, 3)), *block2(384, 256, (3, 3)), *block2(256, 256, (3, 3)),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)), nn.BatchNorm2d(512), nn.ReLU())
            self.netfcaud = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024))
            self.netfclip = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024))
            self.netcnnlip = nn.Sequential(
                *block3(3, 96, (5, 7, 7), (1, 2, 2)), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                *block3(96, 256, (1, 5, 5), (1, 2, 2), (0, 1, 1)),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                *block3(256, 256, (1, 3, 3), p=(0, 1, 1)), *block3(256, 256, (1, 3, 3), p=(0, 1, 1)),
                *block3(256, 256, (1, 3, 3), p=(0, 1, 1)), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                *block3(256, 512, (1, 6, 6)))

        def audio(self, x):
            return self.netfcaud(self.netcnnaud(x).flatten(1))

        def lips(self, x):
            return self.netfclip(self.netcnnlip(x).flatten(1))

    net = S()
    # Tensors only: a weights file is never allowed to run code.
    state = torch.load(os.path.join(folder(), MODEL_FILE), map_location="cpu", weights_only=True)
    # The published file names its tensors as the original module did, under "__S__.".
    net.load_state_dict({key.replace("__S__.", ""): value for key, value in state.items()})
    net.eval()
    _NET["net"] = net
    return net


# --- The face ---

def _size(path, command):
    out = subprocess.run(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                  "-of", "csv=p=0", str(path)]), capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        w, h = (int(x) for x in out.stdout.strip().split(",")[:2])
        return w, h
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return stream.codec_context.width, stream.codec_context.height


def face_square(path, first, last, command=plain, samples=8):
    """(left, top, side) of a square around the largest face, the median over a few frames, or None for no face."""
    import cv2

    w, h = _size(path, command)
    small = 640.0 / max(w, h)
    sw, sh = int(round(w * small)) // 2 * 2, int(round(h * small)) // 2 * 2
    picks = np.linspace(first, max(first, last - 1), samples).astype(int)
    images = []
    for frame in sorted(set(picks)):
        # One seek per frame: older and newer ffmpeg builds agree on that, where frame selection options differ.
        raw = subprocess.run(command(["ffmpeg", "-v", "error", "-ss", "%.4f" % ((frame + 0.5) / float(FPS)), "-i", str(path),
                                      "-an", "-frames:v", "1", "-vf", "scale=%d:%d" % (sw, sh), "-f", "rawvideo",
                                      "-pix_fmt", "bgr24", "-"]), capture_output=True).stdout
        if len(raw) == sw * sh * 3:
            images.append(np.frombuffer(raw, np.uint8).reshape(sh, sw, 3))
    detector = cv2.FaceDetectorYN.create(os.path.join(folder(), FACE_FILE), "", (sw, sh), 0.6)
    found = []
    for image in images:
        _, faces = detector.detect(np.ascontiguousarray(image))
        if faces is not None and len(faces):
            face = max(faces, key=lambda f: f[2] * f[3])
            # YuNet's landmarks: right eye, left eye, nose tip, right and left corner of the mouth.
            eyes = (face[4:6] + face[6:8]) / 2.0
            mouth = (face[10:12] + face[12:14]) / 2.0
            found.append(np.concatenate([eyes, mouth]) / small)
    if len(found) < max(2, len(images) // 2):
        return None
    eyes_x, eyes_y, mouth_x, mouth_y = np.median(np.array(found), axis=0)
    # A face as SyncNet saw them: centred a little below midway between the eyes and the mouth, the square some
    # times the eye-to-mouth distance. When it is larger than the frame it keeps the mouth in the frame.
    span = math.hypot(mouth_x - eyes_x, mouth_y - eyes_y)
    side = int(min(span * CROP, w, h)) // 2 * 2
    cx, cy = (eyes_x + mouth_x) / 2.0, (eyes_y + mouth_y) / 2.0 + LOWER * span
    left = int(min(max(0, cx - side / 2.0), w - side))
    top = int(min(max(0, cy - side / 2.0), h - side))
    return left, top, side


# --- The measure ---

def measure(path, first=0, last=None, command=plain):
    """{"offset", "conf", "dist", "frames"} for frames first..last of a file, with its own sound, or None when there is
    no face, too little to judge, or the measure is not installed."""
    if missing():
        return None
    import torch

    if last is None:
        import av
        with av.open(str(path)) as container:
            last = container.streams.video[0].frames or 0
    first, last = int(first), int(last)
    if last - first < MIN_FRAMES:
        return None
    square = face_square(path, first, last, command)
    if square is None:
        return None
    left, top, side = square
    start, seconds = first / float(FPS), (last - first) / float(FPS)
    raw = subprocess.run(command(["ffmpeg", "-v", "error", "-ss", "%.4f" % start, "-t", "%.4f" % seconds, "-i", str(path),
                                  "-an", "-vf", "fps=%d,crop=%d:%d:%d:%d,scale=224:224" % (SYNC_FPS, side, side, left, top),
                                  "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]), capture_output=True).stdout
    video = np.frombuffer(raw[:len(raw) - len(raw) % (224 * 224 * 3)], np.uint8).reshape(-1, 224, 224, 3)
    sound = subprocess.run(command(["ffmpeg", "-v", "error", "-ss", "%.4f" % start, "-t", "%.4f" % seconds, "-i", str(path),
                                    "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]), capture_output=True).stdout
    samples = np.frombuffer(sound[:len(sound) - len(sound) % 2], "<i2").astype(np.float64)
    if not len(samples) or len(video) < 5:
        return None
    cep = mfcc(samples).T                                   # 13 x F, a column every 10 ms: four to a SyncNet frame
    usable = min(len(video), int(math.floor(len(samples) / 640.0))) - 5
    if usable < MIN_FRAMES - 5:
        return None
    net = network()
    pictures = torch.from_numpy(video.astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0)   # 1 x 3 x T x 224 x 224
    cepstra = torch.from_numpy(cep.astype(np.float32))[None, None]                           # 1 x 1 x 13 x F
    lips, voice = [], []
    with torch.no_grad():
        for at in range(0, usable, 20):
            span = range(at, min(usable, at + 20))
            lips.append(net.lips(torch.cat([pictures[:, :, n:n + 5] for n in span], 0)))
            voice.append(net.audio(torch.cat([cepstra[:, :, :, n * 4:n * 4 + 20] for n in span], 0)))
    lips, voice = torch.cat(lips, 0), torch.cat(voice, 0)
    padded = torch.nn.functional.pad(voice, (0, 0, VSHIFT, VSHIFT))
    width = 2 * VSHIFT + 1
    dists = torch.stack([torch.nn.functional.pairwise_distance(lips[[n], :].repeat(width, 1), padded[n:n + width, :])
                         for n in range(len(lips))], 1)
    mean = dists.mean(1)
    best = int(torch.argmin(mean))
    offset = VSHIFT - best
    return {"offset": int(round(offset * FPS / float(SYNC_FPS))), "conf": round(float(torch.median(mean) - mean[best]), 2),
            "dist": round(float(mean[best]), 2), "frames": last - first}


def judge(found):
    """{"kind", "score", "notes"} from a measure: "in sync", "early", "late" or "no match"."""
    offset, conf = found["offset"], found["conf"]
    sure = min(1.0, max(0.0, conf / 4.0))
    if conf < SURE:
        return {"kind": "no match", "score": int(round(30 * max(0.0, conf) / SURE)),
                "notes": ["the lips do not follow the sound: render the clip again, on another seed"]}
    if abs(offset) <= CLOSE:
        return {"kind": "in sync", "score": int(round(60 + 40 * sure)), "notes": []}
    seconds = abs(offset) / float(FPS)
    kind = "early" if offset < 0 else "late"
    return {"kind": kind, "score": int(round(max(10, 55 - 4 * abs(offset)))),
            "notes": ["the lips move %.2f s %s the sound: render the clip again, on another seed" % (
                seconds, "before" if offset < 0 else "after")]}


def score(path, first=0, last=None, command=plain):
    """The measure and the judgement of frames first..last of a file, or None when it cannot be measured."""
    found = measure(path, first, last, command)
    return None if found is None else dict(judge(found), **found)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    first = int(argv[1]) if len(argv) > 1 else 0
    last = int(argv[2]) if len(argv) > 2 else None
    if missing():
        print("not installed: put %s in %s" % (" and ".join(missing()), folder()))
        return 1
    row = score(argv[0], first, last)
    print(json.dumps(row) if row else "no face, or too short to judge")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
