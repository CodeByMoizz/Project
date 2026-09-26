// The Teachable Machine audio model runs here, in the browser, on the same
// microphone stream. It never receives the Python model's prediction, so the
// two results are produced independently.

var gtmRecognizer = null;
var gtmLabels = [];
var gtmLatestScores = null;
var gtmError = null;

function gtmModelUrls() {
    var base = GTM_SOURCE;

    if (base.indexOf("model.json") !== -1) {
        return {
            model: base,
            metadata: base.replace("model.json", "metadata.json")
        };
    }

    if (base.charAt(base.length - 1) !== "/") {
        base = base + "/";
    }

    return { model: base + "model.json", metadata: base + "metadata.json" };
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

function startGtmListening() {
    if (!gtmRecognizer) {
        return;
    }

    gtmRecognizer.listen(function (result) {
        var scores = {};

        for (var i = 0; i < gtmLabels.length; i++) {
            scores[gtmLabels[i]] = result.scores[i];
        }

        gtmLatestScores = scores;
        return Promise.resolve();
    }, {
        includeSpectrogram: false,
        probabilityThreshold: 0,
        overlapFactor: 0.5,
        invokeCallbackOnNoiseAndUnknown: true
    }).catch(function (error) {
        gtmError = "The GTM model could not use the microphone: " + error.message;
    });
}

function stopGtmListening() {
    if (gtmRecognizer && gtmRecognizer.isListening()) {
        gtmRecognizer.stopListening();
    }

    gtmLatestScores = null;
}

function latestGtmScores() {
    return gtmLatestScores;
}
