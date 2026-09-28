var startButton = document.getElementById("startButton");
var pauseButton = document.getElementById("pauseButton");
var stopButton = document.getElementById("stopButton");

var micDot = document.getElementById("micDot");
var micStatus = document.getElementById("micStatus");
var micStateText = document.getElementById("micStateText");
var gtmState = document.getElementById("gtmState");
var liveMessage = document.getElementById("liveMessage");
var windowCount = document.getElementById("windowCount");

var liveStatus = document.getElementById("liveStatus");
var liveClass = document.getElementById("liveClass");
var livePython = document.getElementById("livePython");
var liveGtm = document.getElementById("liveGtm");
var liveAgreement = document.getElementById("liveAgreement");
var liveSeverity = document.getElementById("liveSeverity");
var liveQuality = document.getElementById("liveQuality");
var liveDifference = document.getElementById("liveDifference");
var liveAlert = document.getElementById("liveAlert");
var liveReviewBox = document.getElementById("liveReviewBox");
var liveReviewReason = document.getElementById("liveReviewReason");
var liveHistory = document.getElementById("liveHistory");

var mediaStream = null;
var audioContext = null;
var processorNode = null;
var sourceNode = null;

var collected = [];
var collectedLength = 0;
var isPaused = false;
var isRunning = false;
var windowsDone = 0;
var historyRows = 0;

function setMicState(text, active) {
    micStatus.textContent = "Microphone: " + text;
    micStateText.textContent = text;

    if (micDot) {
        micDot.style.background = active ? "#238b6c" : "#b4bcc4";
    }
}

function showMessage(text) {
    liveMessage.textContent = text;
}

function updateGtmState() {
    gtmState.textContent = gtmStatusText();
}

function startMonitoring() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setMicState("not supported by this browser", false);
        showMessage("This browser cannot capture microphone audio.");
        return;
    }

    startButton.disabled = true;

    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
        mediaStream = stream;
        isRunning = true;
        isPaused = false;

        setMicState("active", true);
        showMessage("Monitoring. Each window is sent for analysis when it fills.");

        pauseButton.disabled = false;
        stopButton.disabled = false;

        beginCapture(stream);

        loadGtmModel(function (ready) {
            updateGtmState();

            if (ready) {
                startGtmListening();
                updateGtmState();
            }
        });

        stream.getAudioTracks()[0].addEventListener("ended", function () {
            setMicState("disconnected", false);
            stopMonitoring();
        });
    }).catch(function (error) {
        startButton.disabled = false;

        if (error && error.name === "NotAllowedError") {
            setMicState("permission denied", false);
            showMessage("Microphone permission was denied, so monitoring cannot start.");
        } else {
            setMicState("unavailable", false);
            showMessage("The microphone could not be opened: " + error.message);
        }
    });
}

function beginCapture(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(stream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);

    var samplesPerWindow = Math.round(audioContext.sampleRate * WINDOW_SECONDS);

    processorNode.onaudioprocess = function (event) {
        if (!isRunning || isPaused) {
            return;
        }

        var input = event.inputBuffer.getChannelData(0);
        collected.push(new Float32Array(input));
        collectedLength += input.length;

        if (collectedLength >= samplesPerWindow) {
            var windowSamples = joinChunks(collected, collectedLength);
            collected = [];
            collectedLength = 0;
            sendWindow(windowSamples, audioContext.sampleRate);
        }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);
}

function joinChunks(chunks, total) {
    var joined = new Float32Array(total);
    var offset = 0;

    for (var i = 0; i < chunks.length; i++) {
        joined.set(chunks[i], offset);
        offset += chunks[i].length;
    }

    return joined;
}

function makeWavBlob(samples, sampleRate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);

    writeText(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeText(view, 8, "WAVE");
    writeText(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeText(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);

    var offset = 44;
    for (var i = 0; i < samples.length; i++) {
        var value = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
        offset += 2;
    }

    return new Blob([view], { type: "audio/wav" });
}

function writeText(view, offset, text) {
    for (var i = 0; i < text.length; i++) {
        view.setUint8(offset + i, text.charCodeAt(i));
    }
}

function sendWindow(samples, sampleRate) {
    // The GTM scores are read before the request, from the browser's own model,
    // so the server's Python result cannot affect them. They are the per-class
    // maximum over the windows of this segment; the aggregate is then reset so
    // the next segment starts from its own audio.
    var gtmScores = latestGtmScores();
    resetGtmAggregate();

    var blob = makeWavBlob(samples, sampleRate);
    var form = new FormData();
    form.append("audio", blob, "live_window.wav");

    fetch("/api/analyse-window", { method: "POST", body: form })
        .then(function (response) {
            return response.json().then(function (data) {
                return { ok: response.ok, data: data };
            });
        })
        .then(function (result) {
            if (!result.ok) {
                showMessage(result.data.error || "The window could not be analysed.");
                return;
            }

            var detection = result.data.detections[0];
            windowsDone = windowsDone + 1;
            windowCount.textContent = windowsDone;

            showDetection(detection);

            if (gtmScores) {
                sendGtmScores(detection.detection_id, gtmScores);
            }
        })
        .catch(function (error) {
            showMessage("The window could not be sent: " + error.message);
        });
}

function sendGtmScores(detectionId, scores) {
    fetch("/api/gtm/" + detectionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scores: scores })
    })
        .then(function (response) {
            return response.json().then(function (data) {
                return { ok: response.ok, data: data };
            });
        })
        .then(function (result) {
            if (result.ok) {
                showDetection(result.data);
            }
        })
        .catch(function () {
            showMessage("The GTM result could not be sent to the server.");
        });
}

function formatPercent(value) {
    if (value === null || value === undefined) {
        return "—";
    }

    return (value * 100).toFixed(1) + "%";
}

function showDetection(detection) {
    liveStatus.textContent = detection.status || "Classified";
    liveClass.textContent = detection.final_class_name || "—";

    livePython.textContent =
        detection.python_class_name + " " + formatPercent(detection.python_confidence);

    if (detection.gtm_class) {
        liveGtm.textContent =
            detection.gtm_class_name + " " + formatPercent(detection.gtm_confidence);
    } else {
        liveGtm.textContent = "Not received";
    }

    liveAgreement.textContent = detection.agreement_status || "—";
    liveSeverity.textContent = detection.severity || "—";
    liveQuality.textContent = detection.quality || "—";

    if (detection.confidence_difference === null || detection.confidence_difference === undefined) {
        liveDifference.textContent = "—";
    } else {
        liveDifference.textContent = detection.confidence_difference.toFixed(3);
    }

    if (detection.alert_status === "Alert Generated") {
        liveAlert.textContent = detection.final_class_name + " — " + detection.severity;
    } else {
        liveAlert.textContent = "No active alert";
    }

    if (detection.manual_review_required) {
        liveReviewBox.style.display = "block";
        liveReviewReason.textContent = detection.manual_review_reason || "";
    } else {
        liveReviewBox.style.display = "none";
    }

    if (detection.model_message) {
        showMessage(detection.model_message);
    }

    addHistoryRow(detection);
}

function addHistoryRow(detection) {
    var existing = document.getElementById("liveRow" + detection.detection_id);

    if (existing) {
        existing.innerHTML = historyRowHtml(detection);
        return;
    }

    if (historyRows === 0) {
        liveHistory.innerHTML = "";
    }

    var row = document.createElement("tr");
    row.id = "liveRow" + detection.detection_id;
    row.innerHTML = historyRowHtml(detection);
    liveHistory.insertBefore(row, liveHistory.firstChild);
    historyRows = historyRows + 1;
}

function historyRowHtml(detection) {
    var gtmText = detection.gtm_class
        ? detection.gtm_class_name + " " + formatPercent(detection.gtm_confidence)
        : "—";

    var severity = (detection.severity || "low").toLowerCase();

    return (
        "<td>" + new Date().toLocaleTimeString() + "</td>" +
        "<td>" + detection.python_class_name + " " + formatPercent(detection.python_confidence) + "</td>" +
        "<td>" + gtmText + "</td>" +
        "<td>" + (detection.agreement_status || "—") + "</td>" +
        "<td><span class='severity-" + severity + "'>" + (detection.severity || "—") + "</span></td>" +
        "<td>" + (detection.alert_status || "—") + "</td>" +
        "<td><a href='/detection/" + detection.detection_id + "'>Open</a></td>"
    );
}

function pauseMonitoring() {
    if (!isRunning) {
        return;
    }

    isPaused = !isPaused;

    if (isPaused) {
        setMicState("paused", false);
        pauseButton.textContent = "Resume";
        stopGtmListening();
    } else {
        setMicState("active", true);
        pauseButton.textContent = "Pause";
        startGtmListening();
    }

    updateGtmState();
}

function stopMonitoring() {
    isRunning = false;
    isPaused = false;
    collected = [];
    collectedLength = 0;

    stopGtmListening();

    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }

    if (sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(function (track) {
            track.stop();
        });
        mediaStream = null;
    }

    setMicState("stopped", false);
    showMessage("Monitoring stopped. The microphone is released.");

    startButton.disabled = false;
    pauseButton.disabled = true;
    pauseButton.textContent = "Pause";
    stopButton.disabled = true;

    updateGtmState();
}

if (startButton) {
    startButton.addEventListener("click", startMonitoring);
    pauseButton.addEventListener("click", pauseMonitoring);
    stopButton.addEventListener("click", stopMonitoring);
}

window.addEventListener("beforeunload", function () {
    if (isRunning) {
        stopMonitoring();
    }
});

updateGtmState();
