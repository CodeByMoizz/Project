// The Teachable Machine audio model runs here, in the browser, on the same
// microphone stream. It never receives the Python model's prediction, so the
// two results are produced independently.

var gtmRecognizer = null;
var gtmLabels = [];
var gtmError = null;

// The recognizer scores one 1-second window at a time, but a segment or an
// uploaded clip is longer than that. The model was trained on the single
// loudest second of each recording, so the honest question to ask it is "does
// any second of this contain the event", not "what was the last second". Both
// the live page and the upload page therefore keep the highest confidence each
// class reached across the windows of the thing being scored, and the top class
// is the argmax of that vector. Same merge, same policy, both places.
var gtmLiveAggregate = null;

var GTM_LISTEN_OPTIONS = {
    includeSpectrogram: false,
    probabilityThreshold: 0,
    overlapFactor: 0.5,
    invokeCallbackOnNoiseAndUnknown: true
};


function gtmModelUrls() {
    var base = GTM_SOURCE;

    // The speech-commands loader rejects a scheme-less URL, so a project-root
    // path such as "/gtm_model/" has to be resolved against the current page
    // before it is handed over. An absolute GTM_MODEL_URL passes through
    // unchanged.
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

    if (gtmRecognizer) {
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

    if (typeof speechCommands === "undefined") {
        gtmError = "The Teachable Machine library could not be loaded.";
        onReady(false);
        return;
    }

    var urls = gtmModelUrls();

    var recognizer = speechCommands.create(
        "BROWSER_FFT",
        undefined,
        urls.model,
        urls.metadata
    );

    recognizer.ensureModelLoaded().then(function () {
        gtmRecognizer = recognizer;
        gtmLabels = recognizer.wordLabels();
        gtmError = null;
        onReady(true);
    }).catch(function (error) {
        gtmError = "The GTM model could not be loaded: " + error.message;
        onReady(false);
    });
}


// ---------------------------------------------------------------------------
// Aggregation across windows, shared by live monitoring and uploaded clips.
// ---------------------------------------------------------------------------

function newGtmAggregate() {
    return { scores: null, windows: 0 };
}


// Keeps the per-class maximum over every window merged into it.
function mergeGtmWindow(aggregate, windowScores) {
    var scores = aggregate.scores || {};

    for (var i = 0; i < gtmLabels.length; i++) {
        var label = gtmLabels[i];
        var value = windowScores[i];

        if (!(label in scores) || value > scores[label]) {
            scores[label] = value;
        }
    }

    aggregate.scores = scores;
    aggregate.windows = aggregate.windows + 1;
}


function aggregateScores(aggregate) {
    if (!aggregate || !aggregate.scores) {
        return null;
    }

    // A copy, so a later window cannot change a score set already handed out.
    var copy = {};

    for (var label in aggregate.scores) {
        if (aggregate.scores.hasOwnProperty(label)) {
            copy[label] = aggregate.scores[label];
        }
    }

    return copy;
}


// ---------------------------------------------------------------------------
// Live monitoring
// ---------------------------------------------------------------------------

function startGtmListening() {
    if (!gtmRecognizer) {
        return;
    }

    gtmLiveAggregate = newGtmAggregate();

    gtmRecognizer.listen(function (result) {
        mergeGtmWindow(gtmLiveAggregate, result.scores);
        return Promise.resolve();
    }, GTM_LISTEN_OPTIONS).catch(function (error) {
        gtmError = "The GTM model could not use the microphone: " + error.message;
    });
}


function stopGtmListening() {
    if (gtmRecognizer && gtmRecognizer.isListening()) {
        gtmRecognizer.stopListening();
    }

    gtmLiveAggregate = null;
}


// The highest confidence each class reached since the last reset.
function latestGtmScores() {
    return aggregateScores(gtmLiveAggregate);
}


// Called once a segment has taken its scores, so the next segment aggregates
// only its own windows.
function resetGtmAggregate() {
    gtmLiveAggregate = newGtmAggregate();
}


function latestGtmWindowCount() {
    return gtmLiveAggregate ? gtmLiveAggregate.windows : 0;
}


// ---------------------------------------------------------------------------
// Uploaded clips
// ---------------------------------------------------------------------------
//
// An uploaded clip has no microphone behind it, but the model must still see
// the spectrogram the library builds from a real AnalyserNode - recomputing
// that by hand would drift. So the decoded audio is played into a MediaStream
// and the recognizer is pointed at that stream instead of a microphone, which
// is the only injection point the library offers. Everything after the stream
// (AnalyserNode, fftSize, smoothing, frame assembly, the model) is the
// library's own code, unchanged, and the windows are merged with the same
// mergeGtmWindow used live.

var GTM_WINDOW_SECONDS = 1.0;
// The model was trained on the browser's 44100 Hz spectrogram; decoding at any
// other rate would change every bin.
var GTM_SAMPLE_RATE = 44100;
var gtmClipQueue = Promise.resolve();


// Scores the audio at url, or the [startSec, endSec) slice of it, and returns
// {scores, windows} where scores holds the per-class maximum over the windows.
function scoreAudioUrlWithGtm(url, startSec, endSec) {
    function run() {
        return runGtmOnUrl(url, startSec, endSec);
    }

    // Serialised: one recognizer, and listen() can only follow a stopListening().
    gtmClipQueue = gtmClipQueue.then(run, run);

    return gtmClipQueue;
}


function runGtmOnUrl(url, startSec, endSec) {
    if (!gtmRecognizer) {
        return Promise.reject(new Error("The GTM model is not loaded."));
    }

    var context = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: GTM_SAMPLE_RATE
    });
    var destination = context.createMediaStreamDestination();

    if (!navigator.mediaDevices) {
        navigator.mediaDevices = {};
    }

    var realGetUserMedia = navigator.mediaDevices.getUserMedia;
    navigator.mediaDevices.getUserMedia = function () {
        return Promise.resolve(destination.stream);
    };

    function restore() {
        navigator.mediaDevices.getUserMedia = realGetUserMedia;
    }

    var aggregate = newGtmAggregate();
    var clip = null;

    return fetch(url).then(function (response) {
        if (!response.ok) {
            throw new Error("The clip could not be fetched (" + response.status + ").");
        }

        return response.arrayBuffer();
    }).then(function (raw) {
        return context.decodeAudioData(raw);
    }).then(function (decoded) {
        clip = sliceAudioBuffer(context, decoded, startSec, endSec);
        return context.resume();
    }).then(function () {
        return gtmRecognizer.listen(function (result) {
            mergeGtmWindow(aggregate, result.scores);
            return Promise.resolve();
        }, GTM_LISTEN_OPTIONS);
    }).then(function () {
        var source = context.createBufferSource();
        source.buffer = clip;
        source.connect(destination);

        // A short lead lets the extractor settle, and a tail lets the final
        // window covering the end of the clip be scored.
        var lead = 0.3;
        var tail = GTM_WINDOW_SECONDS + 0.3;
        source.start(context.currentTime + lead);

        var waitMs = (lead + source.buffer.duration + tail) * 1000;

        return new Promise(function (resolve) {
            setTimeout(resolve, waitMs);
        });
    }).then(function () {
        return finishGtmClip(context, restore, aggregate);
    }).catch(function (error) {
        return finishGtmClip(context, restore, aggregate).then(function () {
            throw error;
        });
    });
}


function finishGtmClip(context, restore, aggregate) {
    if (gtmRecognizer && gtmRecognizer.isListening()) {
        try {
            gtmRecognizer.stopListening();
        } catch (error) {
            // already stopped
        }
    }

    restore();

    return context.close().catch(function () {
        return null;
    }).then(function () {
        return { scores: aggregateScores(aggregate), windows: aggregate.windows };
    });
}


// Mono-mixed, cut to [startSec, endSec) when those are given, and never shorter
// than the model's window.
function sliceAudioBuffer(context, audioBuffer, startSec, endSec) {
    var rate = audioBuffer.sampleRate;

    var from = 0;
    if (typeof startSec === "number" && startSec > 0) {
        from = Math.min(audioBuffer.length, Math.round(startSec * rate));
    }

    var to = audioBuffer.length;
    if (typeof endSec === "number" && endSec > 0) {
        to = Math.min(audioBuffer.length, Math.round(endSec * rate));
    }

    if (to <= from) {
        to = audioBuffer.length;
        from = 0;
    }

    var wanted = Math.round(GTM_WINDOW_SECONDS * rate);
    var length = Math.max(to - from, wanted);
    var buffer = context.createBuffer(1, length, rate);
    var out = buffer.getChannelData(0);

    for (var channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        var data = audioBuffer.getChannelData(channel);

        for (var i = 0; i < to - from; i++) {
            out[i] = out[i] + data[from + i] / audioBuffer.numberOfChannels;
        }
    }

    return buffer;
}
