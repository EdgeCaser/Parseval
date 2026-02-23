"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

let thumbprints = [];
let selectedThumbprintId = null;

// Per-input file lists (since we can't mutate FileList directly, we track them)
const fileSets = {
  'tp-files': [],
  'analyze-file': [],
  'self-file': [],
};

// ─────────────────────────────────────────────────────────────────────────────
// Logging
// ─────────────────────────────────────────────────────────────────────────────

const logEntries = [];

function log(level, message, detail) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    level,   // 'info' | 'warn' | 'error'
    message,
    detail: detail || null,
  };
  logEntries.unshift(entry);  // newest first
  if (logEntries.length > 200) logEntries.pop();
  renderLogs();
}

function renderLogs() {
  const container = el('log-entries');
  if (!container) return;
  if (logEntries.length === 0) {
    container.innerHTML = '<div style="color:var(--text-muted);font-size:0.8125rem">No activity yet.</div>';
    return;
  }
  container.innerHTML = logEntries.map(e => {
    const color = e.level === 'error' ? '#fca5a5' : e.level === 'warn' ? '#fde68a' : 'var(--text-muted)';
    const detail = e.detail ? `<div style="margin-top:0.2rem;opacity:0.7;white-space:pre-wrap;word-break:break-all">${escapeHtml(e.detail)}</div>` : '';
    return `<div style="margin-bottom:0.6rem;font-size:0.75rem;border-left:3px solid ${color};padding-left:0.5rem">
      <span style="color:${color}">[${e.level.toUpperCase()}]</span>
      <span style="color:var(--text-muted)">${e.time}</span>
      <span style="color:var(--text)"> ${escapeHtml(e.message)}</span>
      ${detail}
    </div>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom modal (replaces alert / confirm)
// ─────────────────────────────────────────────────────────────────────────────

let _modalReject = () => {};

function _showModal({ title, message, buttons }) {
  return new Promise((resolve, reject) => {
    _modalReject = () => { _hideModal(); reject(); };

    el('modal-title').textContent = title || 'Parseval';
    el('modal-message').textContent = message || '';

    const btnContainer = el('modal-buttons');
    btnContainer.innerHTML = '';
    for (const btn of buttons) {
      const b = document.createElement('button');
      b.textContent = btn.label;
      b.className = btn.className || 'modal-btn-ok';
      b.addEventListener('click', () => { _hideModal(); resolve(btn.value); });
      btnContainer.appendChild(b);
    }

    el('modal-overlay').style.display = 'flex';
  });
}

function _hideModal() {
  el('modal-overlay').style.display = 'none';
}

// Alert: single OK button, resolves on dismiss
function showAlert(message, title) {
  return _showModal({
    title: title || 'Parseval',
    message,
    buttons: [{ label: 'OK', className: 'modal-btn-ok', value: true }],
  });
}

// Confirm: Cancel + confirm button, resolves true/rejects on cancel
function showConfirm(message, { title = 'Parseval', confirmLabel = 'OK', danger = false } = {}) {
  return _showModal({
    title,
    message,
    buttons: [
      { label: 'Cancel', className: 'modal-btn-cancel', value: false },
      { label: confirmLabel, className: danger ? 'modal-btn-danger' : 'modal-btn-ok', value: true },
    ],
  }).then(val => {
    if (!val) return Promise.reject();
    return true;
  });
}

// Wire up backdrop click to cancel
document.addEventListener('DOMContentLoaded', () => {
  el('modal-overlay').addEventListener('click', e => {
    if (e.target === el('modal-overlay')) _modalReject();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function scoreToColor(score) {
  const s = Math.max(0, Math.min(1, score));
  const r = s < 0.5 ? 255 : Math.round(255 * (1 - s) * 2);
  const g = s > 0.5 ? 255 : Math.round(255 * s * 2);
  return `rgb(${r},${g},68)`;
}

function scoreToHex(score) {
  const s = Math.max(0, Math.min(1, score));
  const r = s < 0.5 ? 255 : Math.round(255 * (1 - s) * 2);
  const g = s > 0.5 ? 255 : Math.round(255 * s * 2);
  return '#' + [r, g, 68].map(v => v.toString(16).padStart(2, '0')).join('');
}

function formatScore(v) {
  if (v === null || v === undefined) return '—';
  return (v * 100).toFixed(1) + '%';
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  } catch { return iso; }
}

function el(id) { return document.getElementById(id); }

function showError(boxId, message) {
  const box = el(boxId);
  if (box) { box.textContent = message; box.style.display = 'block'; }
}

function clearError(boxId) {
  const box = el(boxId);
  if (box) { box.textContent = ''; box.style.display = 'none'; }
}

function setLoading(isLoading) {
  el('loading-overlay').classList.toggle('visible', isLoading);
  el('results-content').style.display = isLoading ? 'none' : 'block';
}

function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab navigation
// ─────────────────────────────────────────────────────────────────────────────

document.querySelectorAll('.sidebar-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.sidebar-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    el(tab.dataset.panel).classList.add('active');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// File management — removable file lists
// ─────────────────────────────────────────────────────────────────────────────

function bindFileInput(inputId, listId, multiple) {
  const input = el(inputId);
  input.addEventListener('change', function () {
    const newFiles = Array.from(this.files);
    if (multiple) {
      // Merge, deduplicate by name+size
      const existing = fileSets[inputId];
      for (const f of newFiles) {
        if (!existing.some(e => e.name === f.name && e.size === f.size)) {
          existing.push(f);
        }
      }
    } else {
      fileSets[inputId] = newFiles.slice(0, 1);
    }
    // Reset the input so the same file can be re-added after removal
    this.value = '';
    renderFileList(inputId, listId);
  });
}

function renderFileList(inputId, listId) {
  const files = fileSets[inputId];
  const container = el(listId);
  if (!files || files.length === 0) {
    container.innerHTML = '';
    return;
  }
  container.innerHTML = files.map((f, i) => `
    <div class="file-item">
      <span>${escapeHtml(f.name)}</span>
      <span style="color:#555;margin-left:0.3rem">(${(f.size / 1024).toFixed(0)} KB)</span>
      <button class="file-remove-btn" onclick="removeFile('${inputId}', '${listId}', ${i})" title="Remove">✕</button>
    </div>
  `).join('');
}

function removeFile(inputId, listId, index) {
  fileSets[inputId].splice(index, 1);
  renderFileList(inputId, listId);
}

function getFiles(inputId) {
  return fileSets[inputId] || [];
}

bindFileInput('tp-files', 'tp-file-list', true);
bindFileInput('analyze-file', 'analyze-file-label', false);
bindFileInput('self-file', 'self-file-label', false);

// Dragover highlight
document.querySelectorAll('.file-drop').forEach(drop => {
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', () => drop.classList.remove('dragover'));
});

// ─────────────────────────────────────────────────────────────────────────────
// Health check
// ─────────────────────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const resp = await fetch('/api/health');
    const data = await resp.json();
    log('info', 'Health check OK', JSON.stringify(data, null, 2));
    if (!data.spacy_available) {
      el('health-message').textContent =
        `spaCy model '${data.spacy_model}' not found — POS features unavailable. ` +
        `Run: python -m spacy download ${data.spacy_model}`;
      el('health-banner').style.display = 'flex';
      log('warn', `spaCy model '${data.spacy_model}' not available`);
    }
  } catch (e) {
    log('error', 'Health check failed — is the server running?', e.message);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thumbprint management
// ─────────────────────────────────────────────────────────────────────────────

async function loadThumbprints() {
  try {
    const resp = await fetch('/api/thumbprints');
    const data = await resp.json();
    thumbprints = data.thumbprints || [];
    renderThumbprintList();
    updateThumbprintSelect();
    log('info', `Loaded ${thumbprints.length} thumbprint(s)`);
  } catch (e) {
    log('error', 'Failed to load thumbprints', e.message);
  }
}

function renderThumbprintList() {
  const container = el('thumbprint-list');
  if (thumbprints.length === 0) {
    container.innerHTML = `<div class="empty-state"><span class="icon">🗂</span>No thumbprints yet. Create one above.</div>`;
    return;
  }
  container.innerHTML = thumbprints.map(tp => `
    <div class="thumbprint-item ${tp.id === selectedThumbprintId ? 'selected' : ''}"
         data-id="${tp.id}" onclick="selectThumbprint('${tp.id}')">
      <div class="thumbprint-name">${escapeHtml(tp.name)}</div>
      <div class="thumbprint-meta">
        ${tp.paragraph_count} paragraphs · ${tp.source_files.length} file(s) · ${formatDate(tp.created_at)}
      </div>
      <div class="thumbprint-actions">
        <button class="btn btn-danger" onclick="event.stopPropagation(); deleteThumbprint('${tp.id}', '${escapeHtml(tp.name).replace(/'/g, "\\'")}')">
          Delete
        </button>
      </div>
    </div>
  `).join('');
}

function updateThumbprintSelect() {
  const sel = el('analyze-tp-select');
  const current = sel.value;
  sel.innerHTML = '<option value="">— choose a thumbprint —</option>' +
    thumbprints.map(tp =>
      `<option value="${tp.id}" ${tp.id === current ? 'selected' : ''}>${escapeHtml(tp.name)} (${tp.paragraph_count} paras)</option>`
    ).join('');
}

function selectThumbprint(id) {
  selectedThumbprintId = id;
  renderThumbprintList();
  el('analyze-tp-select').value = id;
}

async function createThumbprint() {
  clearError('tp-error');
  const name = el('tp-name').value.trim();
  const files = getFiles('tp-files');

  if (!name) { showError('tp-error', 'Please enter a name for this thumbprint.'); return; }
  if (files.length === 0) { showError('tp-error', 'Please select at least one file.'); return; }

  const btn = el('tp-create-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Building…';
  log('info', `Creating thumbprint "${name}" from ${files.length} file(s)…`);

  const fd = new FormData();
  fd.append('name', name);
  for (const f of files) fd.append('files', f);

  try {
    const resp = await fetch('/api/thumbprints', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      const msg = data.error || 'Failed to create thumbprint.';
      showError('tp-error', msg);
      log('error', 'Thumbprint creation failed', msg);
      return;
    }
    log('info', `Thumbprint "${name}" created (${data.paragraph_count} paragraphs, id: ${data.id})`);
    el('tp-name').value = '';
    fileSets['tp-files'] = [];
    renderFileList('tp-files', 'tp-file-list');
    await loadThumbprints();
  } catch (e) {
    const msg = 'Network error: ' + e.message;
    showError('tp-error', msg);
    log('error', 'Thumbprint creation — network error', e.stack || e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Build Thumbprint';
  }
}

async function deleteThumbprint(id, name) {
  try {
    await showConfirm(`Delete thumbprint "${name}"?\n\nThis cannot be undone.`, {
      title: 'Delete Thumbprint',
      confirmLabel: 'Delete',
      danger: true,
    });
  } catch {
    return; // user cancelled
  }
  log('info', `Deleting thumbprint "${name}"…`);
  try {
    const resp = await fetch(`/api/thumbprints/${id}`, { method: 'DELETE' });
    if (!resp.ok) {
      const data = await resp.json();
      const msg = data.error || 'Failed to delete thumbprint.';
      showAlert(msg, 'Error');
      log('error', 'Delete thumbprint failed', msg);
      return;
    }
    log('info', `Thumbprint "${name}" deleted`);
    if (selectedThumbprintId === id) selectedThumbprintId = null;
    await loadThumbprints();
  } catch (e) {
    log('error', 'Delete thumbprint — network error', e.message);
    showAlert('Network error: ' + e.message, 'Error');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis: Corpus mode
// ─────────────────────────────────────────────────────────────────────────────

async function analyzeCorpus() {
  clearError('analyze-error');
  const tpId = el('analyze-tp-select').value;
  const files = getFiles('analyze-file');

  if (!tpId) { showError('analyze-error', 'Please select a thumbprint.'); return; }
  if (files.length === 0) { showError('analyze-error', 'Please select a document to analyze.'); return; }

  const btn = el('analyze-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing…';
  setLoading(true);
  log('info', `Starting corpus analysis: "${files[0].name}" vs thumbprint ${tpId}…`);

  const fd = new FormData();
  fd.append('thumbprint_id', tpId);
  fd.append('file', files[0]);

  try {
    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      const msg = data.error || 'Analysis failed.';
      showError('analyze-error', msg);
      log('error', 'Corpus analysis failed', msg);
      setLoading(false);
      return;
    }
    log('info', `Corpus analysis complete: ${data.paragraph_count} paragraphs, overall score ${formatScore(data.overall_score)}`);
    renderCorpusResults(data);
  } catch (e) {
    const msg = 'Network error: ' + e.message;
    showError('analyze-error', msg);
    log('error', 'Corpus analysis — network error', e.stack || e.message);
    setLoading(false);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze Document';
    setLoading(false);
  }
}

function renderCorpusResults(data) {
  el('main-header').innerHTML = `
    <h2>Corpus Analysis: ${escapeHtml(data.filename)}</h2>
    <p>Compared against thumbprint: <strong>${escapeHtml(data.thumbprint_name)}</strong> · ${data.paragraph_count} paragraphs</p>
  `;
  const scoreColor = scoreToColor(data.overall_score);
  let html = `
    <div class="overall-score">
      <div class="overall-score-value" style="color:${scoreColor}">${formatScore(data.overall_score)}</div>
      <div>
        <div class="overall-score-label">Overall similarity to thumbprint</div>
        <div class="overall-score-sublabel">Higher = more similar to reference author's style</div>
      </div>
    </div>
    ${scoreLegendHtml('Dissimilar to reference', 'Similar to reference')}
  `;
  html += data.paragraphs.map(p => paragraphBlockHtml(p)).join('');
  el('results-content').innerHTML = html;
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis: Self mode
// ─────────────────────────────────────────────────────────────────────────────

async function analyzeSelf() {
  clearError('self-error');
  const files = getFiles('self-file');

  if (files.length === 0) { showError('self-error', 'Please select a document to analyze.'); return; }

  const btn = el('self-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing…';
  setLoading(true);
  log('info', `Starting self analysis: "${files[0].name}"…`);

  const fd = new FormData();
  fd.append('file', files[0]);

  try {
    const resp = await fetch('/api/analyze/self', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      const msg = data.error || 'Analysis failed.';
      showError('self-error', msg);
      log('error', 'Self analysis failed', msg);
      setLoading(false);
      return;
    }
    log('info', `Self analysis complete: ${data.paragraph_count} paragraphs, consistency ${formatScore(data.overall_consistency)}`);
    renderSelfResults(data);
  } catch (e) {
    const msg = 'Network error: ' + e.message;
    showError('self-error', msg);
    log('error', 'Self analysis — network error', e.stack || e.message);
    setLoading(false);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze for Style Shifts';
    setLoading(false);
  }
}

function renderSelfResults(data) {
  el('main-header').innerHTML = `
    <h2>Self Analysis: ${escapeHtml(data.filename)}</h2>
    <p>Intra-document style consistency · ${data.paragraph_count} paragraphs</p>
  `;
  const consistency = data.overall_consistency;
  const scoreColor = consistency !== null ? scoreToColor(consistency) : '#888';
  let html = `
    <div class="overall-score">
      <div class="overall-score-value" style="color:${scoreColor}">${formatScore(consistency)}</div>
      <div>
        <div class="overall-score-label">Overall style consistency</div>
        <div class="overall-score-sublabel">Low score = style shifts detected · Red paragraphs are outliers</div>
      </div>
    </div>
    ${scoreLegendHtml('Style outlier (suspicious)', 'Consistent with document')}
  `;
  html += data.paragraphs.map(p => paragraphBlockHtml(p)).join('');
  el('results-content').innerHTML = html;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering helpers
// ─────────────────────────────────────────────────────────────────────────────

function scoreLegendHtml(leftLabel, rightLabel) {
  return `
    <div class="score-legend">
      <span>${escapeHtml(leftLabel)}</span>
      <div class="score-gradient"></div>
      <span>${escapeHtml(rightLabel)}</span>
    </div>
  `;
}

function paragraphBlockHtml(p) {
  if (p.is_fragment) {
    return `<div class="paragraph-block fragment">${escapeHtml(p.text)}</div>`;
  }
  const score = p.combined_score;
  const color = scoreToColor(score);
  const bgColor = scoreToHex(score) + '18';
  let tooltipParts = [`Score: ${formatScore(score)}`];
  if (p.style_similarity !== null) tooltipParts.push(`Style: ${formatScore(p.style_similarity)}`);
  if (p.embedding_similarity !== null) tooltipParts.push(`Semantic: ${formatScore(p.embedding_similarity)}`);
  return `
    <div class="paragraph-block" style="border-left-color:${color}; background-color:${bgColor}">
      <div class="paragraph-tooltip">${escapeHtml(tooltipParts.join(' · '))}</div>
      ${escapeHtml(p.text)}
    </div>
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

(async function init() {
  renderLogs();
  await checkHealth();
  await loadThumbprints();
})();
