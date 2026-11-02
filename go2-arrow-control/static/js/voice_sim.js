// Voice maze simulator: a Teachable Machine audio model steers the dog.
// Loaded after shared.js, sim.js and voice_shared.js. sim.js stays unchanged: the functions below
// replace the sim.js functions of the same name (the later script wins). Everything else (maze,
// physics, countdown, leaderboard table, decideCommand) is reused as is.
//
// Replaces: initializeWebcam, loadModelsList, loadSelectedModel, predictFrame, finishRun, loadLeaderboard
// Uses from sim.js: models, currentModel, browserModel, predictionBuffer, currentCommand, runState,
//   runStart, runElapsed, leaderboardRange, todayEntries, AI, resetRun, updateRunButtons,
//   updateTimer, formatTime, renderLeaderboard, updateBestStatus
// Voice does not use decideCommand: it has its own lighter rule (VOICE_RULE in showScores), and one
// recognised word = one step (startStep). predictionBuffer holds the voice votes.
// If you rename any of these in sim.js, update this file too.

const VOICE_MAZE_ID = 'voice-maze-1'; // own leaderboard, see VOICE_MAZE_ID in server/voice.py

// One spoken word = one step. Durations at the simulator's fixed speeds (forward 1.5 cells/s,
// backwards 1.05 cells/s, turning 120°/s): Forward ≈ 3/4 cell, Backwards ≈ 0.4 cell, turns 45°.
const STEP_MS = { Forward: 500, Backwards: 400, Left: 375, Right: 375 };
// Decision rule, much lighter than the camera's: a move needs votesRequired confident results
// among the last `window` results (window 3 at overlapFactor 0.75 = about 0.75 s).
const VOICE_RULE = { minConfidence: 0.75, window: 3, votesRequired: 2 };
// One word shows up in up to ~4 overlapping listen() results: ignore the same move again this soon.
const REPEAT_GUARD_MS = 1000;


let stepTimer = null;
let lastStepCommand = null;
let lastStepAt = 0;

document.addEventListener('DOMContentLoaded', () => {
    initVoiceUploadForm(async (data) => {
        await loadModelsList();
        document.getElementById('model-select').value = data.filename;
        renderModelMapping(data.model);
    });
});

// No camera: the microphone starts when a model is loaded
function initializeWebcam() {
    updateDisplay('Select a voice model, then allow the microphone', 'info');
}

async function loadModelsList() {
    try {
        const res = await fetch('/voice/models');
        const data = await res.json();
        models = data.models || [];
        const select = document.getElementById('model-select');
        const selected = select.value;
        fillModelSelect(select, models);
        if (models.some(m => m.filename === selected)) select.value = selected;
    } catch (e) {
        showToast('Failed to load models list', 'error');
    }
}

// Load the model and start listening, so the bars are live even before Start
async function loadSelectedModel() {
    const filename = document.getElementById('model-select').value;
    const info = models.find(m => m.filename === filename);
    if (!info) {
        showToast('Please select a model first', 'warning');
        return;
    }

    const btn = document.getElementById('load-model-btn');
    btn.disabled = true;
    btn.textContent = 'Loading in browser...';
    try {
        browserModel = await loadVoiceModel(info);
        currentModel = info;
        predictionBuffer = [];
        currentCommand = 'Idle';
        resetRun('ready');
        document.getElementById('model-status').textContent = info.name;
        renderModelMapping(info);
        renderClassBars(info.labels, null);
        updateBestStatus();
        await startVoiceListening(showScores);
        showToast('Model loaded - press Start', 'success');
    } catch (e) {
        console.error(e);
        showToast('Failed to load model (microphone allowed?)', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Load Selected Model';
    }
}

// One listen() result. A move needs VOICE_RULE.votesRequired confident results among the
// last VOICE_RULE.window, which filters out single noise detections.
function showScores(probabilities) {
    if (!currentModel) return;
    renderClassBars(currentModel.labels, probabilities);

    let top = 0;
    probabilities.forEach((p, i) => { if (p > probabilities[top]) top = i; });
    const confidence = probabilities[top];
    // Only confident results vote for a move; everything else votes Idle
    const vote = confidence >= VOICE_RULE.minConfidence ? (currentModel.commands[top] || 'Idle') : 'Idle';

    predictionBuffer.push(vote);
    if (predictionBuffer.length > VOICE_RULE.window) predictionBuffer.shift();
    const votes = predictionBuffer.filter(c => c === vote).length;
    const command = vote !== 'Idle' && votes >= VOICE_RULE.votesRequired ? vote : 'Idle';

    const text = `${currentModel.labels[top]} (${(confidence * 100).toFixed(0)}%)`;
    if (command !== 'Idle') {
        updateDisplay(`${text} -> ${COMMAND_ARROWS[command]} ${command}`, 'success');
    } else if (vote !== 'Idle') {
        updateDisplay(`${text} ${votes}/${VOICE_RULE.votesRequired}`, 'warning'); // heard once, not confirmed yet
    } else {
        updateDisplay(text, 'info');
    }

    if (command !== 'Idle' && runState === 'running') startStep(command);
}


// Move for STEP_MS[command], then stand still. A different word cuts the current step short.
function startStep(command) {
    const now = performance.now();
    if (command === lastStepCommand && now - lastStepAt < REPEAT_GUARD_MS) return;
    lastStepCommand = command;
    lastStepAt = now;

    clearTimeout(stepTimer);
    currentCommand = command;
    stepTimer = setTimeout(() => { currentCommand = 'Idle'; }, STEP_MS[command]);
}


async function finishRun() {
    runState = 'finished';
    currentCommand = 'Idle';
    runElapsed = performance.now() - runStart;
    updateTimer();
    updateRunButtons();

    const el = document.getElementById('sim-countdown');
    el.textContent = '🏁 ' + formatTime(runElapsed);
    el.style.display = 'flex';

    try {
        const res = await fetch('/voice/leaderboard', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ filename: currentModel.filename, time_ms: Math.round(runElapsed) })
        });
        const data = await res.json();
        if (!res.ok) {
            showToast(data.error || 'Could not save the time', 'error');
        } else if (data.new_best_today) {
            showToast(`New best for ${data.name}: ${formatTime(data.time_ms)}!`, 'success');
        } else {
            showToast(`${formatTime(data.time_ms)} - best today stays ${formatTime(data.best_today_ms)}`, 'info');
        }
    } catch (e) {
        showToast('Could not save the time', 'error');
    }
    loadLeaderboard();
}

async function loadLeaderboard() {
    try {
        const [shown, today] = await Promise.all([
            fetch(`/api/leaderboard?range=${leaderboardRange}&maze_id=${VOICE_MAZE_ID}`).then(r => r.json()),
            leaderboardRange === 'today' ? null : fetch(`/api/leaderboard?range=today&maze_id=${VOICE_MAZE_ID}`).then(r => r.json()),
        ]);
        todayEntries = (today || shown).entries;
        renderLeaderboard(shown.entries);
        updateBestStatus();
    } catch (e) {
        console.error('Leaderboard failed', e);
    }
}
