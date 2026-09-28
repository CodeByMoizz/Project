// The browser model. A small CNN on log-mel spectrograms, loaded with
// tf.loadLayersModel from gtm_model/. It never sees the Python model's
// prediction, so the two results stay independent.

var gtmModel = null;
var gtmLabels = [];
var gtmError = null;
var gtmVersion = "";

var GTM_HOP = 0.5;

var gtmLiveAggregate = null;


function gtmModelUrls() {
    var base = GTM_SOURCE;

    // A scheme-less path such as "/gtm_model/" has to be resolved against the
    // page before it is loaded. An absolute GTM_MODEL_URL passes through.
    function absolute(url) {
        return new URL(url, window.location.href).href;
    }

    if (base.indexOf("model.json") !== -1) {
        return {
            model: absolute(base),
            metadata: absolute(base.replace("model.json", "metadata.json"))
        };
    }

    if (base.charAt(base.length - 1) !== "/") {
        base = base + "/";
    }

    return { model: absolute(base + "model.json"), metadata: absolute(base + "metadata.json") };
}


function gtmStatusText() {
    if (gtmError) {
        return gtmError;
    }

    if (!GTM_READY) {
        return "Not configured";
    }

    if (gtmModel) {
        return "Listening in browser";
    }

    return "Loading in browser…";
}


function loadGtmModel(onReady) {
    if (!GTM_READY) {
        gtmError = "Not configured";
        onReady(false);
        return;
    }

    if (typeof tf === "undefined") {
        gtmError = "TensorFlow.js could not be loaded.";
        onReady(false);
        return;
    }

    var urls = gtmModelUrls();

    fetch(urls.metadata).then(function (r) {
        if (!r.ok) {
            throw new Error("metadata.json is missing (" + r.status + ")");
        }
        return r.json();
    }).then(function (meta) {
        gtmLabels = meta.labels || [];
        gtmVersion = meta.version || "";

        if (gtmLabels.length !== GTM_CLASSES.length) {
            throw new Error("metadata.json has " + gtmLabels.length +
                " labels, the app expects " + GTM_CLASSES.length);
        }

        for (var i = 0; i < gtmLabels.length; i++) {
            if (gtmLabels[i] !== GTM_CLASSES[i]) {
                throw new Error("label " + i + " is '" + gtmLabels[i] +
                    "' but the app expects '" + GTM_CLASSES[i] + "'");
            }
        }

        return tf.loadLayersModel(urls.model);
    }).then(function (model) {
        gtmModel = model;
        gtmError = null;
        onReady(true);
    }).catch(function (error) {
        gtmError = "The GTM model could not be loaded: " + error.message;
        onReady(false);
    });
}


// One window in, ten scores out.
function gtmPredict(samples) {
    return tf.tidy(function () {
        var f = lmFeatures(samples);
        var batch = f.reshape([1, LM_FRAMES, LM_MELS, 1]);
        return gtmModel.predict(batch).dataSync();
    });
}


// ---------------------------------------------------------------------------
// Aggregation across windows, shared by live monitoring and uploaded clips.
// ---------------------------------------------------------------------------

function newGtmAggregate() {
    return { rows: [], windows: 0 };
}


function mergeGtmWindow(aggregate, windowScores) {
    var row = [];

    for (var i = 0; i < gtmLabels.length; i++) {
        row.push(windowScores[i]);
    }

    aggregate.rows.push(row);
    aggregate.windows = aggregate.windows + 1;
}


// Mean over the windows of the clip or segment. Chosen on the validation split:
// mean beat max and top-3 mean on macro F1. Max let any window that faintly
// resembled an event outvote the null class.
function aggregateScores(aggregate) {
    if (!aggregate || !aggregate.rows.length) {
        return null;
    }

    var scores = {};

    for (var i = 0; i < gtmLabels.length; i++) {
        var total = 0;

        for (var r = 0; r < aggregate.rows.length; r++) {
            total = total + aggregate.rows[r][i];
        }

        scores[gtmLabels[i]] = total / aggregate.rows.length;
    }

    return scores;
}


// ---------------------------------------------------------------------------
// Live monitoring
// ---------------------------------------------------------------------------

var gtmContext = null;
var gtmStream = null;
var gtmNode = null;
var gtmRing = null;
var gtmFilled = 0;
var gtmSinceHop = 0;


function startGtmListening() {
    if (!gtmModel || gtmContext) {
        return;
    }

    gtmRing = new Float32Array(LM_WIN);
    gtmFilled = 0;
    gtmSinceHop = 0;
    gtmLiveAggregate = newGtmAggregate();

    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
        gtmStream = stream;
        gtmContext = new AudioContext({ sampleRate: LM_SR });

        var source = gtmContext.createMediaStreamSource(stream);
        gtmNode = gtmContext.createScriptProcessor(4096, 1, 1);

        gtmNode.onaudioprocess = function (event) {
            var input = event.inputBuffer.getChannelData(0);

            // Slide the ring buffer along by the block we just received.
            gtmRing.copyWithin(0, input.length);
            gtmRing.set(input, LM_WIN - input.length);

            gtmFilled = Math.min(LM_WIN, gtmFilled + input.length);
            gtmSinceHop = gtmSinceHop + input.length;

            if (gtmFilled >= LM_WIN && gtmSinceHop >= LM_SR * GTM_HOP) {
                gtmSinceHop = 0;
                mergeGtmWindow(gtmLiveAggregate, gtmPredict(gtmRing));
            }
        };

        source.connect(gtmNode);
        gtmNode.connect(gtmContext.destination);
    }).catch(function (error) {
        gtmError = "The GTM model could not use the microphone: " + error.message;
    });
}


function stopGtmListening() {
    if (gtmNode) {
        gtmNode.onaudioprocess = null;
        gtmNode.disconnect();
        gtmNode = null;
    }

    if (gtmStream) {
        gtmStream.getTracks().forEach(function (t) { t.stop(); });
        gtmStream = null;
    }

    if (gtmContext) {
        gtmContext.close();
        gtmContext = null;
    }

    gtmLiveAggregate = null;
}


function latestGtmScores() {
    return aggregateScores(gtmLiveAggregate);
}


function resetGtmAggregate() {
    gtmLiveAggregate = newGtmAggregate();
}


function latestGtmWindowCount() {
    return gtmLiveAggregate ? gtmLiveAggregate.windows : 0;
}


// ---------------------------------------------------------------------------
// Uploaded clips
// ---------------------------------------------------------------------------

// Scores the audio at url, or the [startSec, endSec) slice of it, and returns
// {scores, windows} where scores holds the per-class mean over the windows.
function scoreAudioUrlWithGtm(url, startSec, endSec) {
    if (!gtmModel) {
        return Promise.reject(new Error("The GTM model is not loaded."));
    }

    var ctx = new AudioContext({ sampleRate: LM_SR });

    return fetch(url).then(function (r) {
        if (!r.ok) {
            throw new Error("The clip could not be fetched (" + r.status + ").");
        }
        return r.arrayBuffer();
    }).then(function (raw) {
        return ctx.decodeAudioData(raw);
    }).then(function (buffer) {
        var samples = mono(buffer);
        samples = slice(samples, startSec, endSec);

        var count = windowCount(samples.length);
        var aggregate = newGtmAggregate();

        lmWindows(samples, count).forEach(function (w) {
            mergeGtmWindow(aggregate, gtmPredict(w));
        });

        return ctx.close().catch(function () { return null; }).then(function () {
            return { scores: aggregateScores(aggregate), windows: aggregate.windows };
        });
    }).catch(function (error) {
        return ctx.close().catch(function () { return null; }).then(function () {
            throw error;
        });
    });
}


function mono(buffer) {
    var out = new Float32Array(buffer.length);

    for (var c = 0; c < buffer.numberOfChannels; c++) {
        var d = buffer.getChannelData(c);
        for (var i = 0; i < d.length; i++) {
            out[i] = out[i] + d[i] / buffer.numberOfChannels;
        }
    }

    return out;
}


function slice(samples, startSec, endSec) {
    var from = 0;
    if (typeof startSec === "number" && startSec > 0) {
        from = Math.min(samples.length, Math.round(startSec * LM_SR));
    }

    var to = samples.length;
    if (typeof endSec === "number" && endSec > 0) {
        to = Math.min(samples.length, Math.round(endSec * LM_SR));
    }

    if (to <= from) {
        return samples;
    }

    return samples.subarray(from, to);
}


function windowCount(length) {
    if (length <= LM_WIN) {
        return 1;
    }

    return Math.round((length - LM_WIN) / (LM_SR * GTM_HOP)) + 1;
}
