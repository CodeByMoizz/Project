// Same log-mel features as feature_extraction/logmel.py. If these two drift the
// model sees something it was never trained on, so they are checked against
// each other before training.

var LM_SR = 16000;
var LM_WIN = 16000;
var LM_FRAME = 1024;
var LM_STEP = 512;
var LM_FFT = 1024;
var LM_MELS = 64;
var LM_FMIN = 20;
var LM_FMAX = 8000;
var LM_EPS = 1e-6;
var LM_FRAMES = 31;
var LM_PAD = 16384;

var lmMel = null;


function lmHzToMel(hz) {
    return 1127.0 * Math.log(1.0 + hz / 700.0);
}


// Same triangles as tf.signal.linear_to_mel_weight_matrix, including the
// dropped DC bin.
function lmMelMatrix() {
    if (lmMel) {
        return lmMel;
    }

    var bins = LM_FFT / 2 + 1;
    var nyquist = LM_SR / 2.0;

    var lower = lmHzToMel(LM_FMIN);
    var upper = lmHzToMel(LM_FMAX);

    var edges = [];
    for (var i = 0; i < LM_MELS + 2; i++) {
        edges.push(lower + (upper - lower) * i / (LM_MELS + 1));
    }

    var flat = new Float32Array(bins * LM_MELS);

    for (var b = 1; b < bins; b++) {
        var hz = nyquist * b / (bins - 1);
        var m = lmHzToMel(hz);

        for (var j = 0; j < LM_MELS; j++) {
            var lo = edges[j];
            var mid = edges[j + 1];
            var hi = edges[j + 2];

            var up = (m - lo) / (mid - lo);
            var down = (hi - m) / (hi - mid);
            var v = Math.min(up, down);

            flat[b * LM_MELS + j] = v > 0 ? v : 0;
        }
    }

    // kept, so no enclosing tf.tidy can dispose the cached matrix
    lmMel = tf.keep(tf.tensor2d(flat, [bins, LM_MELS]));
    return lmMel;
}


// samples: Float32Array of one window. Returns a [31, 64] tensor.
function lmFeatures(samples) {
    var mel = lmMelMatrix();

    return tf.tidy(function () {
        var padded = new Float32Array(LM_PAD);
        var n = Math.min(samples.length, LM_WIN);

        for (var i = 0; i < n; i++) {
            padded[i] = samples[i];
        }

        var x = tf.tensor1d(padded);
        var spec = tf.signal.stft(x, LM_FRAME, LM_STEP, LM_FFT);
        var mag = tf.abs(spec);
        var out = tf.log(tf.add(tf.matMul(mag, mel), tf.scalar(LM_EPS)));

        var moments = tf.moments(out);
        var std = tf.sqrt(moments.variance);
        out = tf.div(tf.sub(out, moments.mean), tf.add(std, tf.scalar(LM_EPS)));

        return tf.slice(out, [0, 0], [LM_FRAMES, LM_MELS]);
    });
}


// Evenly spaced 1 second windows, matching windows() in the Python module.
function lmWindows(samples, count) {
    if (samples.length <= LM_WIN) {
        var one = new Float32Array(LM_WIN);
        one.set(samples.subarray(0, Math.min(samples.length, LM_WIN)));
        return [one];
    }

    var starts = [];
    var last = samples.length - LM_WIN;

    if (count === 1) {
        starts.push(Math.floor(last / 2));
    } else {
        for (var i = 0; i < count; i++) {
            starts.push(Math.floor(last * i / (count - 1)));
        }
    }

    return starts.map(function (s) {
        return samples.subarray(s, s + LM_WIN);
    });
}
