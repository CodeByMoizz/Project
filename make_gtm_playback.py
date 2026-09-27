import os
import sys
import numpy as np
import soundfile as sf

SR = 44100
BATCH = 20
BATCHES_PER_CLASS = 3
IN_ROOT = "gtm_upload"
OUT_ROOT = "gtm_playback"


def main():
    if not os.path.isdir(IN_ROOT):
        print("no %s folder - run make_gtm_clips.py first" % IN_ROOT)
        sys.exit(1)

    os.makedirs(OUT_ROOT, exist_ok=True)
    classes = sorted(d for d in os.listdir(IN_ROOT) if os.path.isdir(os.path.join(IN_ROOT, d)))

    for cls in classes:
        files = sorted(f for f in os.listdir(os.path.join(IN_ROOT, cls)) if f.endswith(".wav"))
        if not files:
            continue
        step = max(1, len(files) // (BATCH * BATCHES_PER_CLASS))
        picked = files[::step][:BATCH * BATCHES_PER_CLASS]

        made = 0
        for b in range(BATCHES_PER_CLASS):
            group = picked[b * BATCH:(b + 1) * BATCH]
            if len(group) < BATCH:
                break
            out = []
            for f in group:
                y, _ = sf.read(os.path.join(IN_ROOT, cls, f), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                y = y[:SR]
                if len(y) < SR:
                    y = np.pad(y, (0, SR - len(y)))
                out.append(y)
            montage = np.concatenate(out)
            name = "%s_batch%d.wav" % (cls, b + 1)
            sf.write(os.path.join(OUT_ROOT, name), montage, SR, subtype="PCM_16")
            made += 1

        print("%-24s %d batches of %d sec" % (cls, made, BATCH))

    print()
    print("play one file while Teachable Machine records 20 seconds on the matching class.")


if __name__ == "__main__":
    main()
