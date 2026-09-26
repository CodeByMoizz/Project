var audioInput = document.getElementById("audioInput");
var audioPreview = document.getElementById("audioPreview");
var fileChosen = document.getElementById("fileChosen");

function showChosenFiles() {
    var files = audioInput.files;

    if (!files || files.length === 0) {
        fileChosen.textContent = "No file chosen yet.";
        return;
    }

    if (files.length === 1) {
        fileChosen.textContent = files[0].name;
    } else {
        fileChosen.textContent = files.length + " files selected.";
    }

    if (audioPreview) {
        audioPreview.src = URL.createObjectURL(files[0]);
        audioPreview.load();
    }
}

if (audioInput) {
    audioInput.addEventListener("change", showChosenFiles);
}
