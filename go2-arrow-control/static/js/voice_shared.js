// Shared by the voice pages (voice_sim.js and voice_control.js).
// Needs vendor/tf.min.js, vendor/speech-commands.min.js and js/shared.js.

// listen() options. Each result covers ~1 s of audio; a new one arrives every (1 - overlapFactor) s.
const LISTEN_OPTIONS = {
    includeSpectrogram: false,
    probabilityThreshold: 0,              // pass every result on; the usual threshold and consensus apply
    invokeCallbackOnNoiseAndUnknown: true,
    overlapFactor: 0.85                    // increase for faster reactions
};

// Teachable Machine audio model (speech-commands recognizer) in this browser, one at a time
let loadedRecognizer = null;
let loadedRecognizerFilename = null;

// Load a voice model (info from /voice/models). Cached until another model is loaded.
async function loadVoiceModel(info) {
    if (!info) return null;
    if (loadedRecognizer && loadedRecognizerFilename === info.filename) return loadedRecognizer;

    await stopVoiceListening();
    // speech-commands only accepts absolute http(s) URLs for the metadata
    const recognizer = speechCommands.create('BROWSER_FFT', undefined,
        new URL(info.model_url, location.href).href,
        new URL(info.metadata_url, location.href).href);
    await recognizer.ensureModelLoaded();
    loadedRecognizer = recognizer;
    loadedRecognizerFilename = info.filename;
    return recognizer;
}

// Start the microphone (the browser asks for permission the first time).
// onScores(probabilities) gets one number per class, in the order of info.labels (= wordLabels).
async function startVoiceListening(onScores) {
    await stopVoiceListening();
    await loadedRecognizer.listen(result => onScores(Array.from(result.scores)), LISTEN_OPTIONS);
}

async function stopVoiceListening() {
    if (loadedRecognizer && loadedRecognizer.isListening()) {
        await loadedRecognizer.stopListening();
    }
}

// Upload form #voice-upload-form (#model-name, #model-file). Same as initUploadForm() in shared.js,
// which is tied to #upload-form and /upload_model. onUploaded(data) runs after a successful upload.
function initVoiceUploadForm(onUploaded) {
    const form = document.getElementById('voice-upload-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const nameInput = document.getElementById('model-name');
        const fileInput = document.getElementById('model-file');
        const submitBtn = form.querySelector('button[type="submit"]');

        const formData = new FormData();
        formData.append('model_name', nameInput.value);
        formData.append('model', fileInput.files[0]);

        submitBtn.disabled = true;
        submitBtn.textContent = 'Uploading...';

        try {
            const res = await fetch('/voice/upload_model', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (res.ok) {
                const warnings = data.model && data.model.warnings.length;
                showToast(warnings ? 'Model uploaded, check the class names' : 'Model uploaded!', warnings ? 'warning' : 'success');
                form.reset();
                await onUploaded(data);
            } else {
                showToast(data.error, 'error');
            }
        } catch (e) {
            showToast('Upload failed', 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Upload Model';
        }
    });
}
