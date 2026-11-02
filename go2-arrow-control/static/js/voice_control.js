// Robot control with a Teachable Machine audio model.
// Loaded after shared.js, control.js and voice_shared.js. control.js stays unchanged: the functions
// below replace the control.js functions of the same name (the later script wins). Pilot handling,
// settings, status polling and emergency stop are reused as is. Instead of webcam frames
// (sendFrame), each listen() result is sent to /voice/submit_prediction (sendScores), which turns
// every confirmed word into one fixed robot step (see ROBOT_STEPS in server/voice.py).
//
// Replaces: initializeWebcam, loadModelsList, ensureBrowserModel, loadSelectedModel,
//   startInference, stopInference, updateButtons
// Uses from control.js: currentState, browserModel, frameInFlight, setCurrentModel, resetClassBars
//   and the element ids of control.html
// If you rename any of these in control.js, update this file too.

document.addEventListener('DOMContentLoaded', () => {
    initVoiceUploadForm(async (data) => {
        await loadModelsList();
        document.getElementById('model-select').value = data.filename;
        renderModelMapping(data.model);
    });
});

// No camera: the microphone starts with Start Control
function initializeWebcam() {
    updateDisplay('Microphone starts with ▶️ Start Control', 'info');
}

async function loadModelsList() {
    const select = document.getElementById('model-select');
    try {
        const res = await fetch('/voice/models');
        const data = await res.json();

        if (data.models) {
            fillModelSelect(select, data.models);

            // Show the model that is currently loaded (shared by all groups)
            const current = data.models.find(m => m.filename === data.current) || null;
            if (current) select.value = current.filename;
            setCurrentModel(current);
        }
    } catch (e) {
        showToast('Failed to load models list', 'error');
    }
}

async function ensureBrowserModel() {
    browserModel = await loadVoiceModel(currentState.currentModel);
}

async function loadSelectedModel() {
    const select = document.getElementById('model-select');
    const filename = select.value;

    if (!filename) {
        showToast('Please select a model first', 'warning');
        return;
    }

    const btn = document.getElementById('load-model-btn');
    btn.disabled = true;
    btn.textContent = 'Loading...';

    try {
        const formData = new FormData();
        formData.append('filename', filename);

        const res = await fetch('/voice/load_model', {
            method: 'POST',
            body: formData
        });

        if (res.ok) {
            const data = await res.json();
            setCurrentModel(data.model);
            btn.textContent = 'Loading in browser...';
            await ensureBrowserModel();
            showToast('Model loaded successfully', 'success');
            updateDisplay('Model Ready - Press Start', 'success');
        } else {
            const err = await res.json();
            showToast(`Error: ${err.error}`, 'error');
        }
    } catch (e) {
        console.error(e);
        showToast('Failed to load model', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Load Selected Model';
    }
}

async function startInference() {
    if (!currentState.modelLoaded || !currentState.isPilot) return;

    const startBtn = document.getElementById('start-btn');
    startBtn.disabled = true;
    startBtn.textContent = 'Starting...';
    updateDisplay('Starting control...', 'info');

    try {
        await ensureBrowserModel();

        const res = await fetch('/start_inference', {
            method: 'POST'
        });
        const data = await res.json().catch(() => ({}));

        if (!res.ok || !data.success) {
            const message = data.error || data.message || 'Failed to start control';
            showToast(message, 'error');
            updateDisplay('Start failed', 'error');
            return;
        }

        currentState.inferenceActive = true;
        await startVoiceListening(sendScores);
        updateDisplay('Listening...', 'success');
        showToast('Control started', 'success');
    } catch (e) {
        console.error(e);
        if (currentState.inferenceActive) stopInference();
        showToast('Failed to start control (microphone allowed?)', 'error');
        updateDisplay('Start failed', 'error');
    } finally {
        startBtn.textContent = '▶️ Start Control';
        updateButtons();
    }
}

function stopInference(notifyServer = true) {
    stopVoiceListening();
    currentState.inferenceActive = false;
    updateButtons();
    updateDisplay('Stopped', 'warning');
    resetClassBars();

    if (notifyServer) {
        fetch('/stop_inference', { method: 'POST' });
    }
}

// One listen() result -> server. Like sendFrame() in control.js, without the camera image.
async function sendScores(probabilities) {
    if (!currentState.isPilot) {
        stopInference(false);
        return;
    }
    if (!currentState.inferenceActive || frameInFlight) return;
    frameInFlight = true;

    renderClassBars(currentState.currentModel.labels, probabilities);
    const start = Date.now();

    try {
        const res = await fetch('/voice/submit_prediction', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ probabilities })
        });

        const data = await res.json();
        if (data.model_changed) loadModelsList();
        document.getElementById('latency-val').textContent = Date.now() - start;

        if (res.ok && currentState.inferenceActive) {
            const conf = (data.confidence * 100).toFixed(0);
            const cmd = data.command_to_execute;

            let text = `${data.prediction} (${conf}%)`;
            if (cmd && cmd !== 'Idle') {
                text += ` -> EXECUTE: ${cmd}`;
                document.getElementById('last-cmd').textContent = cmd;
            }

            updateDisplay(text, cmd !== 'Idle' ? 'success' : 'info');
        }
    } catch (e) {
        console.error(e);
    } finally {
        frameInFlight = false;
    }
}

// Same as control.js, but keeps #voice-upload-form usable for non-pilots
function updateButtons() {
    const isPilot = currentState.isPilot;

    document.getElementById('start-btn').disabled = !currentState.modelLoaded || currentState.inferenceActive || !isPilot;
    document.getElementById('stop-btn').disabled = !currentState.inferenceActive || !isPilot;
    document.getElementById('emergency-btn').disabled = !isPilot;
    document.getElementById('load-model-btn').disabled = !isPilot;
    document.getElementById('save-settings-btn').disabled = !isPilot;

    const inputs = document.querySelectorAll('.model-panel input, .model-panel select');
    inputs.forEach(input => {
        if (!input.closest('#voice-upload-form')) {
            input.disabled = !isPilot;
        }
    });
}
