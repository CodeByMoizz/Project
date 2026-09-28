import os
import sys
import numpy as np
import librosa
import soundfile as sf

SR = 44100
WIN = SR
SEG_PER_CLIP = 1

SPLITS = ["training", "validation"]
IN_ROOT = "audio_dataset"
OUT_ROOT = "gtm_upload"


def load_windows(path, want):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    n = len(y) // WIN
    if n == 0:
        return []
    chunks = []
    for i in range(n):
        seg = y[i * WIN:(i + 1) * WIN]
        chunks.append((float(np.sqrt(np.mean(seg ** 2))), i, seg))
    chunks.sort(key=lambda c: c[0], reverse=True)
    keep = [c for c in chunks if c[0] > 0.005][:want]
    if not keep:
        keep = chunks[:1]
    keep.sort(key=lambda c: c[1])
    return keep


def main():
    if not os.path.isdir(IN_ROOT):
        print("run this from the project root (no audio_dataset folder here)")
        sys.exit(1)

    classes = set()
    for split in SPLITS:
        d = os.path.join(IN_ROOT, split)
        if os.path.isdir(d):
            classes.update(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    classes = sorted(classes)
    if not classes:
        print("no class folders found in training/ or validation/ - run prepare_dataset.py first")
        sys.exit(1)

    for cls in classes:
        out_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)
        for old in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, old))

        files = []
        for split in SPLITS:
            d = os.path.join(IN_ROOT, split, cls)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                        files.append(os.path.join(d, f))

        written = 0
        skipped = 0
        for path in files:
            try:
                windows = load_windows(path, SEG_PER_CLIP)
            except Exception:
                skipped += 1
                continue
            audio_id = os.path.splitext(os.path.basename(path))[0]
            for _, idx, seg in windows:
                name = "%s__seg%02d.wav" % (audio_id, idx)
                sf.write(os.path.join(out_dir, name), seg.astype(np.float32), SR, subtype="PCM_16")
                written += 1

        print("%-24s %4d clips -> %4d samples%s" % (cls, len(files), written,
              "  (%d unreadable)" % skipped if skipped else ""))

    print()
    print("done. upload each folder under %s/ into the matching class in Teachable Machine." % OUT_ROOT)


if __name__ == "__main__":
    main()
