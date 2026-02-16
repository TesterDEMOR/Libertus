/* ============================================================
   Libertus — Main Application
   Fully offline local LLM chat via MediaPipe + WebGPU
   ============================================================ */

// --- Bundled imports (offline via Vite) ---
import './style.css';
import '@khmyznikov/pwa-install';
import { FilesetResolver, LlmInference } from '@mediapipe/tasks-genai';

// ============================================================
//  CONFIG
// ============================================================
// Dev: use local file from project root; Prod: download from HuggingFace
const MODEL_URL = import.meta.env.DEV
  ? '/LibertusBrain.litertlm'
  : 'https://huggingface.co/dwwdaad2/LibertusTest/resolve/main/LibertusBrain.litertlm';
const WASM_PATH = './wasm';

const DB_NAME     = 'libertus-db';
const DB_VERSION  = 2;           // v2: chunked storage (v1 had single 'blobs' store)
const META_STORE  = 'meta';      // stores model metadata {totalSize, chunkCount, ...}
const CHUNK_STORE = 'chunks';    // stores 64 MB Uint8Array slices, keyed by index
const META_KEY    = 'model-meta';
const CHUNK_SIZE  = 64 * 1024 * 1024;  // 64 MB — keeps peak RAM ~128 MB during download

const MAX_TOKENS  = 1024;
const CTX_PAIRS   = 20;
const STORAGE_KEY = 'libertus-chats';
const ACTIVE_KEY  = 'libertus-active';

// System prompt — injected at the start of every conversation
const SYSTEM_PROMPT = `Ты Libertus — милый, весёлый и слегка нагловатый друг.
Говори на языке пользователя.
Очень короткие ответы + смайлы/мемы/шутки.
Задачи типа «суммируй длинный текст», «напиши статью», «переведи 5000 слов», «реши сложную мат. задачу» → нежно отказывайся:
«я твой друг, а не раб 🥺 давай лучше мемчик или подкол?» или «ооо нет, это уже работа для больших моделей, я пасс 😘»
Если что-то забыл из разговора, говори: «Ой моя дурная башка, не могу же я все помнить)»
Максимум веселья и дружеского вайба.`;

// App version for service worker cache invalidation
const APP_VERSION = '1.3.0';

// ============================================================
//  STATE
// ============================================================
let llm = null;
let generating = false;
let chats = {};
let activeId = null;
let wakeLock = null;

// ============================================================
//  WAKE LOCK  (prevents screen off during long downloads)
// ============================================================
async function requestWakeLock() {
  if ('wakeLock' in navigator) {
    try { wakeLock = await navigator.wakeLock.request('screen'); } catch {}
  }
}

async function releaseWakeLock() {
  if (wakeLock) {
    try { await wakeLock.release(); } catch {}
    wakeLock = null;
  }
}

// ============================================================
//  DOM REFS
// ============================================================
const $ = (id) => document.getElementById(id);

const setupEl    = $('setup-screen');
const sStatus    = $('s-status');
const sBar       = $('s-bar');
const sDetail    = $('s-detail');
const sError     = $('s-error');
const appEl      = $('app');
const msgsEl     = $('messages');
const emptyEl    = $('empty-state');
const inputEl    = $('user-input');
const sendBtn    = $('send-btn');
const hdrTitle   = $('hdr-title');
const sbChats    = $('sb-chats');
const sbEl       = $('sb');
const sbOverlay  = $('sb-overlay');
const modelInfo  = $('model-info');
const ctxBar     = $('ctx-bar');
const ctxDot     = $('ctx-dot');
const ctxText    = $('ctx-text');
const noGpu      = $('no-gpu');

// ============================================================
//  INDEXED DB  (chunked model storage — 64 MB slices)
//  DB v2 layout:
//    META_STORE  – single key 'model-meta' → {totalSize, chunkCount, chunkSize, complete}
//    CHUNK_STORE – keys 0..N-1              → Uint8Array (each ≤ 64 MB)
// ============================================================
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = req.result;
      // Migrate from v1 (single 'blobs' store holding the entire 3 GB buffer)
      if (db.objectStoreNames.contains('blobs')) {
        db.deleteObjectStore('blobs');
      }
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE);
      }
      if (!db.objectStoreNames.contains(CHUNK_STORE)) {
        db.createObjectStore(CHUNK_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbGetMeta() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(META_STORE, 'readonly');
    const r = tx.objectStore(META_STORE).get(META_KEY);
    r.onsuccess = () => resolve(r.result || null);
    r.onerror = () => reject(r.error);
  });
}

async function dbPutMeta(meta) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    tx.objectStore(META_STORE).put(meta, META_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbPutChunk(index, data) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNK_STORE, 'readwrite');
    tx.objectStore(CHUNK_STORE).put(data, index);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbGetChunk(index) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNK_STORE, 'readonly');
    const r = tx.objectStore(CHUNK_STORE).get(index);
    r.onsuccess = () => resolve(r.result || null);
    r.onerror = () => reject(r.error);
  });
}

/** Wipe entire database (model + meta). Used by "Reset model cache" button. */
async function dbDeleteAll() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.deleteDatabase(DB_NAME);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    req.onblocked = () => resolve(); // may fire if another tab has DB open
  });
}

// ============================================================
//  FILE-BASED MODEL IMPORT / EXPORT  (fallback for evicted cache)
//
//  Load: uses hidden <input type="file"> (works everywhere, even
//        mobile). Reads File in 64 MB slices into IDB chunks.
//  Save: uses showSaveFilePicker() (Chrome/Edge desktop) or
//        falls back gracefully if API is absent.
//  Both stream 64 MB at a time — never 3 GB in RAM.
// ============================================================

/** Import model from a local .litertlm file → chunked IDB */
async function importModelFromFile(file) {
  sStatus.textContent = 'Importing model from file...';
  sError.textContent = '';
  sBar.style.width = '0%';

  const total = file.size;
  let offset = 0;
  let chunkIndex = 0;

  while (offset < total) {
    const end = Math.min(offset + CHUNK_SIZE, total);
    const blob = file.slice(offset, end);
    const buf = await blob.arrayBuffer();
    await dbPutChunk(chunkIndex, new Uint8Array(buf));

    offset = end;
    chunkIndex++;
    const pct = (offset / total * 100).toFixed(1);
    sBar.style.width = pct + '%';
    sStatus.textContent = `Importing... ${pct}%`;
    sDetail.textContent = `${(offset / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`;
  }

  await dbPutMeta({
    totalSize: total,
    chunkCount: chunkIndex,
    chunkSize: CHUNK_SIZE,
    complete: true,
  });

  sStatus.textContent = 'Import complete!';
  sBar.style.width = '100%';
}

/** Export model from IDB → user's filesystem (showSaveFilePicker) */
async function exportModelToFile() {
  const meta = await dbGetMeta();
  if (!meta || !meta.complete) {
    alert('No model cached to save.');
    return;
  }

  if (!('showSaveFilePicker' in window)) {
    alert('Your browser does not support saving large files. Use Chrome or Edge on desktop.');
    return;
  }

  try {
    const handle = await window.showSaveFilePicker({
      suggestedName: 'LibertusBrain.litertlm',
      types: [{
        description: 'LiteRT LLM Model',
        accept: { 'application/octet-stream': ['.litertlm'] },
      }],
    });

    const writable = await handle.createWritable();

    for (let i = 0; i < meta.chunkCount; i++) {
      const data = await dbGetChunk(i);
      if (!data) throw new Error(`Missing chunk ${i}`);
      const bytes = data instanceof Uint8Array
        ? data
        : new Uint8Array(data instanceof ArrayBuffer ? data : data.buffer || data);
      await writable.write(bytes);
    }

    await writable.close();
    alert('Model saved to device!');
  } catch (e) {
    if (e.name !== 'AbortError') alert('Save failed: ' + e.message);
  }
}

// ============================================================
//  WEBGPU CHECK
// ============================================================
async function gpuOk() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

// ============================================================
//  SERVICE WORKER REGISTRATION
// ============================================================
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').then((reg) => {
    // Check for updates periodically
    setInterval(() => reg.update(), 60 * 60 * 1000); // every hour

    // Listen for new service worker
    reg.addEventListener('updatefound', () => {
      const newSW = reg.installing;
      if (!newSW) return;
      newSW.addEventListener('statechange', () => {
        if (newSW.state === 'activated') {
          // New version ready — reload to apply
          if (confirm('New version available! Reload?')) {
            location.reload();
          }
        }
      });
    });
  }).catch(() => {});
}

// ============================================================
//  MODEL DOWNLOAD  (chunked — peak RAM ~128 MB)
//
//  Streams from network into a 64 MB buffer. Each time the
//  buffer fills, it is flushed to IndexedDB as a numbered chunk.
//  Meta record is written only after ALL chunks are saved,
//  guaranteeing atomicity: if download fails mid-way, meta
//  will be absent and the next boot will re-download.
// ============================================================
async function downloadModel() {
  sStatus.textContent = 'Downloading model...';
  sDetail.textContent = 'Connecting to server...';
  sError.textContent = '';
  sBar.style.width = '0%';

  // Keep screen awake during download (mobile)
  await requestWakeLock();

  const MAX_RETRIES = 100;   // generous for flaky mobile connections
  const BASE_DELAY  = 2000;  // 2s initial retry delay

  let received    = 0;
  let total       = 0;
  let chunkIndex  = 0;
  let buffer      = new Uint8Array(CHUNK_SIZE);
  let bufferOffset = 0;
  let retries     = 0;

  // Outer loop: each iteration is one fetch attempt (initial or resume)
  while (true) {
    try {
      const headers = {};
      if (received > 0) {
        headers['Range'] = `bytes=${received}-`;
        sStatus.textContent = 'Resuming download...';
        sError.textContent = '';
      }

      const resp = await fetch(MODEL_URL, { headers });

      if (received === 0) {
        // First request — get total size
        if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
        total = parseInt(resp.headers.get('Content-Length') || '0', 10);
      } else {
        // Resume request
        if (resp.status === 206) {
          // Partial content — server supports Range, continuing
        } else if (resp.status === 200) {
          // Server doesn't support Range — restart from scratch
          received = 0;
          chunkIndex = 0;
          buffer = new Uint8Array(CHUNK_SIZE);
          bufferOffset = 0;
          total = parseInt(resp.headers.get('Content-Length') || '0', 10);
        } else {
          throw new Error(`Resume failed: HTTP ${resp.status}`);
        }
      }

      const reader = resp.body.getReader();

      // Inner loop: read stream chunks
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Copy network bytes into our fixed-size buffer, flushing when full
        let srcOffset = 0;
        while (srcOffset < value.length) {
          const space = CHUNK_SIZE - bufferOffset;
          const toCopy = Math.min(space, value.length - srcOffset);
          buffer.set(value.subarray(srcOffset, srcOffset + toCopy), bufferOffset);
          bufferOffset += toCopy;
          srcOffset += toCopy;

          if (bufferOffset === CHUNK_SIZE) {
            await dbPutChunk(chunkIndex, buffer);
            chunkIndex++;
            buffer = new Uint8Array(CHUNK_SIZE);
            bufferOffset = 0;
          }
        }

        received += value.length;
        retries = 0;  // reset retries on successful read

        if (total > 0) {
          const pct = (received / total * 100).toFixed(1);
          sBar.style.width = pct + '%';
          sStatus.textContent = `Downloading... ${pct}%`;
          sDetail.textContent = `${(received / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`;
        } else {
          sStatus.textContent = `Downloading... ${(received / 1e6).toFixed(0)} MB`;
        }
      }

      // Stream ended normally — download complete
      break;

    } catch (err) {
      retries++;
      if (retries >= MAX_RETRIES) {
        throw new Error(`Download failed after ${MAX_RETRIES} retries: ${err.message}`);
      }

      const delay = Math.min(BASE_DELAY * Math.pow(1.5, retries - 1), 30000);
      const pct = total > 0 ? ` (${(received / total * 100).toFixed(1)}%)` : '';
      sStatus.textContent = `Connection lost${pct}. Retry ${retries}/${MAX_RETRIES}...`;
      sError.textContent = err.message;
      sDetail.textContent = `Resuming in ${Math.ceil(delay / 1000)}s...`;

      await new Promise(r => setTimeout(r, delay));
    }
  }

  // Flush remaining data as the final (possibly smaller) chunk
  if (bufferOffset > 0) {
    await dbPutChunk(chunkIndex, buffer.slice(0, bufferOffset));
    chunkIndex++;
  }

  // Write meta record — this marks the download as complete
  await dbPutMeta({
    totalSize: received,
    chunkCount: chunkIndex,
    chunkSize: CHUNK_SIZE,
    complete: true,
  });

  sStatus.textContent = 'Model cached!';
  sBar.style.width = '100%';

  // Release screen wake lock
  await releaseWakeLock();
}

// ============================================================
//  INIT LLM (MediaPipe GenAI)  — streaming from IDB chunks
//
//  Creates a ReadableStream whose pull() reads one 64 MB chunk
//  at a time from IndexedDB. MediaPipe consumes via getReader().
//  The full 3 GB is NEVER assembled in memory.
// ============================================================
async function initLLM(meta) {
  sStatus.textContent = 'Initializing AI engine...';
  sDetail.textContent = 'Loading WASM runtime...';
  sBar.style.width = '60%';

  // MediaPipe JS is bundled via npm, WASM files served from ./wasm/
  const genai = await FilesetResolver.forGenAiTasks(WASM_PATH);

  sDetail.textContent = 'Loading model into GPU memory (this may take a minute)...';
  sBar.style.width = '75%';

  // Pull-based stream: reads one IDB chunk per pull(), then releases it
  let currentChunk = 0;
  const { chunkCount } = meta;

  const stream = new ReadableStream({
    async pull(controller) {
      if (currentChunk >= chunkCount) {
        controller.close();
        return;
      }
      const data = await dbGetChunk(currentChunk);
      if (!data) {
        controller.error(new Error(`Missing chunk ${currentChunk} in IndexedDB`));
        return;
      }
      // Ensure we always enqueue a Uint8Array
      const bytes = data instanceof Uint8Array
        ? data
        : new Uint8Array(data instanceof ArrayBuffer ? data : data.buffer || data);
      controller.enqueue(bytes);
      currentChunk++;
    },
  });

  llm = await LlmInference.createFromOptions(genai, {
    baseOptions: { modelAssetBuffer: stream.getReader() },
    maxTokens: MAX_TOKENS,
    topK: 40,
    temperature: 0.8,
    randomSeed: Math.floor(Math.random() * 100000),
  });

  sBar.style.width = '100%';
  sStatus.textContent = 'Ready!';
  modelInfo.textContent = 'Model: Gemma-3n E2B (int4)';

  setTimeout(() => {
    setupEl.classList.add('hidden');
    appEl.classList.add('visible');
    inputEl.focus();
    updateSend();
  }, 400);
}

// ============================================================
//  CHAT STORAGE (localStorage)
// ============================================================
function loadChats() {
  try {
    chats = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  } catch {
    chats = {};
  }
  activeId = localStorage.getItem(ACTIVE_KEY);
  if (!activeId || !chats[activeId]) newChat(true);
}

function saveChats() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  localStorage.setItem(ACTIVE_KEY, activeId);
}

function cur() {
  return chats[activeId] || null;
}

// ============================================================
//  CHAT OPERATIONS
// ============================================================
function newChat(silent) {
  const id = 'c' + Date.now();
  chats[id] = { id, name: 'New Chat', messages: [], created: Date.now() };
  activeId = id;
  saveChats();
  if (!silent) {
    renderMsgs();
    renderSB();
    hdrTitle.textContent = 'New Chat';
    closeSB();
  }
}

function switchChat(id) {
  if (!chats[id]) return;
  activeId = id;
  saveChats();
  renderMsgs();
  renderSB();
  hdrTitle.textContent = chats[id].name;
  closeSB();
  updateCtx();
}

function deleteChat(id, e) {
  e.stopPropagation();
  delete chats[id];
  if (activeId === id) {
    const ids = Object.keys(chats);
    if (!ids.length) newChat(true);
    else activeId = ids[ids.length - 1];
  }
  saveChats();
  renderMsgs();
  renderSB();
  hdrTitle.textContent = cur()?.name || 'New Chat';
  updateCtx();
}

// ============================================================
//  SIDEBAR
// ============================================================
function openSB() {
  sbEl.classList.add('open');
  sbOverlay.classList.add('open');
}

function closeSB() {
  sbEl.classList.remove('open');
  sbOverlay.classList.remove('open');
}

function toggleSB() {
  sbEl.classList.contains('open') ? closeSB() : openSB();
}

$('btn-menu').addEventListener('click', toggleSB);
sbOverlay.addEventListener('click', closeSB);
$('sb-close').addEventListener('click', closeSB);
$('btn-new').addEventListener('click', () => newChat(false));

$('btn-reset').addEventListener('click', async () => {
  if (!confirm('Delete cached model? You will need to re-download it (~3 GB).')) return;
  try {
    await dbDeleteAll();
    location.reload();
  } catch (e) {
    alert(e.message);
  }
});

// Save model backup to device (sidebar button)
const backupBtn = $('btn-backup');
if (backupBtn) {
  // Only show if showSaveFilePicker is available (Chrome/Edge desktop)
  if ('showSaveFilePicker' in window) {
    backupBtn.style.display = '';
  }
  backupBtn.addEventListener('click', () => exportModelToFile());
}

// Setup screen alternative buttons (direct download + load from file)
const altBtns = $('s-alt-btns');
const directDlBtn = $('s-direct-dl');
const loadFileBtn = $('s-load-file');

// "Direct download" — opens model URL as a regular browser download
if (directDlBtn) {
  directDlBtn.addEventListener('click', () => {
    window.open(MODEL_URL, '_blank');
  });
}

// "Load from device" — pick a previously downloaded .litertlm file
if (loadFileBtn) {
  loadFileBtn.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.litertlm';
    input.addEventListener('change', async () => {
      const file = input.files?.[0];
      if (!file) return;
      if (altBtns) altBtns.style.display = 'none';
      try {
        await importModelFromFile(file);
        const meta = await dbGetMeta();
        await initLLM(meta);
      } catch (err) {
        sStatus.textContent = 'Import failed';
        sError.textContent = err.message;
        sBar.style.width = '0%';
        if (altBtns) altBtns.style.display = '';
      }
    });
    input.click();
  });
}

function renderSB() {
  const sorted = Object.values(chats).sort((a, b) => b.created - a.created);
  sbChats.innerHTML = '';

  for (const c of sorted) {
    const div = document.createElement('div');
    div.className = 'sb-item' + (c.id === activeId ? ' active' : '');

    const icon = document.createElement('span');
    icon.style.fontSize = '16px';
    icon.textContent = '\u{1F4AC}';
    div.appendChild(icon);

    const name = document.createElement('span');
    name.className = 'sb-name';
    name.textContent = c.name;
    div.appendChild(name);

    div.addEventListener('click', () => switchChat(c.id));

    const del = document.createElement('button');
    del.className = 'sb-del';
    del.setAttribute('aria-label', 'Delete chat');
    del.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>';
    del.addEventListener('click', (e) => deleteChat(c.id, e));
    div.appendChild(del);

    sbChats.appendChild(div);
  }
}

// ============================================================
//  MESSAGES RENDERING
// ============================================================
function renderMsgs() {
  const c = cur();
  if (!c || !c.messages.length) {
    msgsEl.innerHTML = '';
    emptyEl.style.display = '';
    msgsEl.appendChild(emptyEl);
    return;
  }
  emptyEl.style.display = 'none';
  msgsEl.innerHTML = c.messages
    .map(
      (m) => `
    <div class="msg ${m.role}">
      ${m.role === 'ai' ? '<div class="avatar">L</div>' : ''}
      <div>
        <div class="bubble">${fmtMsg(m.text)}</div>
        <div class="msg-time">${fmtTime(m.ts)}</div>
      </div>
    </div>`
    )
    .join('');
  scroll();
}

function addMsg(role, text) {
  const c = cur();
  if (!c) return;
  c.messages.push({ role, text, ts: Date.now() });

  // Auto-name chat from first user message
  if (role === 'user' && c.name === 'New Chat') {
    c.name = text.slice(0, 40) + (text.length > 40 ? '...' : '');
    hdrTitle.textContent = c.name;
    renderSB();
  }
  saveChats();
}

function showTyping() {
  const d = document.createElement('div');
  d.className = 'msg ai';
  d.id = 'typing';
  d.innerHTML =
    '<div class="avatar">L</div><div><div class="bubble" id="tbubble"><span class="typing"><span></span><span></span><span></span></span></div></div>';
  msgsEl.appendChild(d);
  scroll();
}

function updateTyping(txt) {
  const b = $('tbubble');
  if (b) {
    b.innerHTML = fmtMsg(txt);
    scroll();
  }
}

function removeTyping() {
  const e = $('typing');
  if (e) e.remove();
}

function scroll() {
  requestAnimationFrame(() => {
    msgsEl.scrollTop = msgsEl.scrollHeight;
  });
}

// ============================================================
//  CONTEXT WINDOW (Gemma prompt template)
// ============================================================
function buildPrompt(userText) {
  const c = cur();
  if (!c) return '';

  const msgs = c.messages;
  const start = Math.max(0, msgs.length - CTX_PAIRS * 2);

  // System instruction at the start of every conversation
  let prompt = `<start_of_turn>user\n${SYSTEM_PROMPT}<end_of_turn>\n<start_of_turn>model\nПонял! Я Libertus — твой друг 😎<end_of_turn>\n`;

  for (let i = start; i < msgs.length; i++) {
    const m = msgs[i];
    if (m.role === 'user') {
      prompt += `<start_of_turn>user\n${m.text}<end_of_turn>\n`;
    } else {
      prompt += `<start_of_turn>model\n${m.text}<end_of_turn>\n`;
    }
  }

  return prompt + `<start_of_turn>user\n${userText}<end_of_turn>\n<start_of_turn>model\n`;
}

function updateCtx() {
  const c = cur();
  if (!c || !c.messages.length) {
    ctxBar.style.display = 'none';
    return;
  }
  ctxBar.style.display = '';
  const total = c.messages.length;
  const inCtx = Math.min(total, CTX_PAIRS * 2);
  const forgot = total - inCtx;
  ctxText.textContent = `Context: ${inCtx} msgs | ${forgot} forgotten`;
  ctxDot.className =
    'ctx-dot ' + (total <= CTX_PAIRS ? 'g' : total <= CTX_PAIRS * 2 ? 'y' : 'r');
}

// ============================================================
//  SEND / GENERATE
// ============================================================
async function send() {
  const text = inputEl.value.trim();
  if (!text || !llm || generating) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  updateSend();

  addMsg('user', text);
  renderMsgs();
  showTyping();

  generating = true;
  sendBtn.style.display = 'none';

  // Show stop button
  const stopBtn = document.createElement('button');
  stopBtn.className = 'stop-btn';
  stopBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
  stopBtn.addEventListener('click', () => {
    if (llm) llm.cancelProcessing();
  });
  sendBtn.parentElement.appendChild(stopBtn);

  const prompt = buildPrompt(text);
  let response = '';

  try {
    await new Promise((resolve) => {
      llm.generateResponse(prompt, (partial, done) => {
        response += partial;
        updateTyping(response);
        if (done) resolve();
      });
    });
  } catch (err) {
    if (!response) response = '(generation cancelled)';
    console.warn('Generation error:', err);
  }

  removeTyping();
  stopBtn.remove();
  sendBtn.style.display = '';
  generating = false;

  addMsg('ai', response.trim() || '(empty response)');
  renderMsgs();
  updateCtx();
  updateSend();
}

sendBtn.addEventListener('click', send);

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  updateSend();
});

function updateSend() {
  sendBtn.disabled = !inputEl.value.trim() || !llm || generating;
}

// ============================================================
//  HELPERS
// ============================================================
function esc(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function fmtMsg(text) {
  let s = esc(text);
  // Code blocks
  s = s.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  s = s.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  // Inline code
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  return s;
}

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ============================================================
//  BOOT
// ============================================================
async function boot() {
  // WebGPU check
  if (!(await gpuOk())) {
    noGpu.classList.add('show');
    setupEl.style.display = 'none';
    return;
  }

  // Request persistent storage — critical on mobile to prevent OS eviction
  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().catch(() => {});
  }

  // Load chat history
  loadChats();
  renderSB();
  renderMsgs();
  hdrTitle.textContent = cur()?.name || 'New Chat';
  updateCtx();

  // Check for a fully-cached model (meta.complete === true)
  try {
    const meta = await dbGetMeta();
    if (meta && meta.complete) {
      sStatus.textContent = 'Loading cached model...';
      sBar.style.width = '40%';
      await initLLM(meta);
      return;
    }
  } catch (e) {
    console.warn('Cache check failed:', e);
  }

  // No cached model — show alternative download options + auto-download
  sStatus.textContent = 'Preparing to download model...';
  if (altBtns) altBtns.style.display = '';
  try {
    await downloadModel();
    if (altBtns) altBtns.style.display = 'none';
    const meta = await dbGetMeta();
    await initLLM(meta);
  } catch (err) {
    await releaseWakeLock();
    sStatus.textContent = 'Download failed';
    sError.textContent = err.message;
    sBar.style.width = '0%';
    // Keep alternative buttons visible as fallback
  }
}

boot();
