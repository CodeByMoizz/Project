// Runs the Teachable Machine model over the segments of a clip that has just
// been uploaded, so an upload gets a GTM result the same way live monitoring
// does. The scoring policy is the shared one in gtm.js: every window the
// recognizer emits for a segment is merged with mergeGtmWindow, and the class
// is the argmax of the per-class maximum.
//
// The Python model has already run on the server by the time this page renders,
// and its scores are never read here, so the two results stay independent.

var gtmUploadStatus = document.getElementById("gtmUploadStatus");


function setGtmUploadStatus(text) {
    if (gtmUploadStatus) {
        gtmUploadStatus.textContent = text;
    }
}


function scoreUploadedSegments() {
    if (typeof GTM_PENDING === "undefined" || !GTM_PENDING.length) {
        return;
    }

    if (!GTM_READY) {
        setGtmUploadStatus("Not configured");
        return;
    }

    setGtmUploadStatus("Loading the model in the browser…");

    loadGtmModel(function (ok) {
        if (!ok) {
            setGtmUploadStatus(gtmError || "The GTM model could not be loaded.");
            return;
        }

        var done = 0;
        var failed = 0;

        function report() {
            var text = done + " of " + GTM_PENDING.length + " segment(s) scored";

            if (failed) {
                text = text + ", " + failed + " failed";
            }

            setGtmUploadStatus(text);
        }

        report();

        GTM_PENDING.forEach(function (segment) {
            scoreAudioUrlWithGtm(segment.audio_url, segment.start_sec, segment.end_sec)
                .then(function (result) {
                    if (!result.scores) {
                        throw new Error("no window was scored");
                    }

                    return sendUploadGtmScores(segment.detection_id, result.scores);
                })
                .then(function () {
                    done = done + 1;
                    report();
                })
                .catch(function (error) {
                    failed = failed + 1;
                    report();
                    console.warn("GTM scoring failed for detection "
                        + segment.detection_id + ": " + error.message);
                });
        });
    });
}


function sendUploadGtmScores(detectionId, scores) {
    return fetch("/api/gtm/" + detectionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scores: scores })
    }).then(function (response) {
        if (!response.ok) {
            throw new Error("the server rejected the scores (" + response.status + ")");
        }

        return response.json();
    });
}


scoreUploadedSegments();
