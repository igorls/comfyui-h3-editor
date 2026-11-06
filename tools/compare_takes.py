"""Measure how closely a span of one take matches a span of another. Needs av and numpy.

python compare_takes.py A.mp4 a_start B.mp4 b_start frames
Prints picture PSNR for the span, per third, and audio correlation over it.
"""

import sys

import av
import numpy as np


def frames(path):
    with av.open(path) as box:
        return np.stack([f.to_ndarray(format="rgb24") for f in box.decode(video=0)]).astype(np.float32)


def sound(path):
    with av.open(path) as box:
        if not box.streams.audio:
            return None, 0
        stream = box.streams.audio[0]
        rate = stream.rate
        parts = [f.to_ndarray() for f in box.decode(audio=0)]
    wave = np.concatenate(parts, axis=-1)
    if wave.ndim == 2 and wave.shape[0] > 2:
        wave = wave.T
    return wave.reshape(-1, wave.shape[-1]).mean(axis=0).astype(np.float32), rate


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    a_path, a_start, b_path, b_start, count = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    a = frames(a_path)
    b = frames(b_path)
    print("frames: A %s  B %s  size %s" % (a.shape[0], b.shape[0], a.shape[1:3]))
    x = a[a_start:a_start + count]
    y = b[b_start:b_start + count]
    print("span PSNR %.2f dB over %s frames" % (psnr(x, y), len(x)))
    third = max(1, len(x) // 3)
    print("thirds PSNR", [round(psnr(x[i:i + third], y[i:i + third]), 2) for i in range(0, third * 3, third)])
    if b_start + count < b.shape[0]:
        after = b[b_start + count:b_start + count + 12]
        step = np.abs(b[b_start + count - 1] - b[b_start + count]).mean()
        inner = np.mean([np.abs(b[i] - b[i + 1]).mean() for i in range(b_start, b_start + count - 1)])
        later = np.mean([np.abs(after[i] - after[i + 1]).mean() for i in range(len(after) - 1)]) if len(after) > 1 else 0
        print("seam jump %.2f  inside hold %.2f  after seam %.2f (mean abs per frame)" % (step, inner, later))
    wa, ra = sound(a_path)
    wb, rb = sound(b_path)
    if wa is None or wb is None or ra != rb:
        print("audio: not comparable", ra, rb)
        return
    sa, sb, n = int(a_start / 24 * ra), int(b_start / 24 * rb), int(count / 24 * ra)
    pa, pb = wa[sa:sa + n], wb[sb:sb + n]
    n = min(len(pa), len(pb))
    pa, pb = pa[:n], pb[:n]
    print("audio correlation %.4f  rms A %.4f  rms B %.4f" % (
        float(np.corrcoef(pa, pb)[0, 1]) if n and pa.std() and pb.std() else 0.0,
        float(np.sqrt(np.mean(pa ** 2))), float(np.sqrt(np.mean(pb ** 2)))))


if __name__ == "__main__":
    main()
