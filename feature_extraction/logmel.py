import numpy as np
import tensorflow as tf

SR = 16000
WIN = 16000
FRAME = 1024
STEP = 512
FFT = 1024
MELS = 64
FMIN = 20
FMAX = 8000
EPS = 1e-6
FRAMES = 31

# 16000 samples give 30 frames, the model wants 31, so the window is zero
# padded to 16384 before the stft. Same padding in the browser.
PAD = 16384

_mel = None


def mel_matrix():
    global _mel
    if _mel is None:
        _mel = tf.signal.linear_to_mel_weight_matrix(
            MELS, FFT // 2 + 1, SR, FMIN, FMAX
        ).numpy()
    return _mel


def logmel(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    x = x[:WIN]
    x = np.pad(x, (0, PAD - WIN))

    s = tf.signal.stft(x, frame_length=FRAME, frame_step=STEP, fft_length=FFT)
    mag = tf.abs(s).numpy()
    mel = mag.dot(mel_matrix())
    out = np.log(mel + EPS)

    out = (out - out.mean()) / (out.std() + EPS)
    return out.astype(np.float32)[:FRAMES]


def windows(y, count):
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if len(y) <= WIN:
        return [np.pad(y, (0, WIN - len(y)))]

    if count == 1:
        starts = [(len(y) - WIN) // 2]
    else:
        starts = np.linspace(0, len(y) - WIN, count).astype(int)

    return [y[s:s + WIN] for s in starts]
