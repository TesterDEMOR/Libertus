import './style.css';
import '@khmyznikov/pwa-install';
import { FilesetResolver, LlmInference } from '@mediapipe/tasks-genai';

const MODEL_URL = import.meta.env.DEV
  ? '/LibertusBrain.litertlm'
  : 'https://huggingface.co/dwwdaad2/LibertusTest/resolve/main/LibertusBrain.litertlm';
const WASM_PATH = './wasm';

const DB_NAME = 'libertus-db';
const DB_VERSION = 2;
const META_STORE = 'meta';
const CHUNK_STORE = 'chunks';
const META_KEY = 'model-meta';
const CHUNK_SIZE = 64 * 1024 * 1024;

const MAX_TOKENS = 1024;
const CTX_PAIRS = 3;
const STORAGE_KEY = 'libertus-chats';
const ACTIVE_KEY = 'libertus-active';

const SYSTEM_PROMPT = `Ты Libertus.
Твой создатель — Бер Максим (Академия МВД РК).
Твоя цель — душевное общение.
На сложные задачи (статьи, код, математика) вежливо отказывай: "Я сегодня на расслабоне, давай лучше просто поболтаем? 🙃"
Если что-то забыл — честно скажи "Ой, вылетело из головы".
help me`;

const APP_VERSION = '1.5.3';

let llm = null;
let generating = false;
let chats = {};
let activeId = null;
let wakeLock = null;
let downloadController = null;
let isDownloading = false;

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

const $ = (id) => document.getElementById(id);

const setupEl = $('setup-screen');
const sStatus = $('s-status');
const sBar = $('s-bar');
const sDetail = $('s-detail');
const sError = $('s-error');
const appEl = $('app');
const msgsEl = $('messages');
const emptyEl = $('empty-state');
const inputEl = $('user-input');
const sendBtn = $('send-btn');
const hdrTitle = $('hdr-title');
const sbChats = $('sb-chats');
const sbEl = $('sb');
const sbOverlay = $('sb-overlay');
const modelInfo = $('model-info');
const ctxBar = $('ctx-bar');
const ctxDot = $('ctx-dot');
const ctxText = $('ctx-text');
const noGpu = $('no-gpu');

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = req.result;
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

async function dbDeleteAll() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.deleteDatabase(DB_NAME);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    req.onblocked = () => resolve();
  });
}

async function importModelFromFile(file) {
  if (isDownloading && downloadController) {
    downloadController.abort();
    isDownloading = false;
    await releaseWakeLock();
  }

  sStatus.textContent = 'Importing model from file...';
  sError.textContent = '';
  sBar.style.width = '0%';
  if (altBtns) altBtns.style.display = 'none';

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

async function gpuOk() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').then((reg) => {
    setInterval(() => reg.update(), 60 * 60 * 1000);
    reg.addEventListener('updatefound', () => {
      const newSW = reg.installing;
      if (!newSW) return;
      newSW.addEventListener('statechange', () => {
        if (newSW.state === 'activated') {
          if (confirm('New version available! Reload?')) {
            location.reload();
          }
        }
      });
    });
  }).catch(() => {});
}

async function downloadModel() {
  isDownloading = true;
  downloadController = new AbortController();

  sStatus.textContent = 'Downloading model...';
  sDetail.textContent = 'Connecting to server...';
  sError.textContent = '';
  sBar.style.width = '0%';

  await requestWakeLock();

  const MAX_RETRIES = 100;
  const BASE_DELAY = 2000;

  let received = 0;
  let total = 0;
  let chunkIndex = 0;
  let buffer = new Uint8Array(CHUNK_SIZE);
  let bufferOffset = 0;
  let retries = 0;

  while (true) {
    if (downloadController.signal.aborted) {
        isDownloading = false;
        throw new Error('Download aborted');
    }

    try {
      const headers = {};
      if (received > 0) {
        headers['Range'] = `bytes=${received}-`;
        sStatus.textContent = 'Resuming download...';
        sError.textContent = '';
      }

      const resp = await fetch(MODEL_URL, { headers, signal: downloadController.signal });

      if (received === 0) {
        if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
        total = parseInt(resp.headers.get('Content-Length') || '0', 10);
      } else {
        if (resp.status === 206) {
        } else if (resp.status === 200) {
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

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

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
        retries = 0;

        if (total > 0) {
          const pct = (received / total * 100).toFixed(1);
          sBar.style.width = pct + '%';
          sStatus.textContent = `Downloading... ${pct}%`;
          sDetail.textContent = `${(received / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`;
        } else {
          sStatus.textContent = `Downloading... ${(received / 1e6).toFixed(0)} MB`;
        }
      }

      break;

    } catch (err) {
      if (downloadController.signal.aborted || err.name === 'AbortError') {
          isDownloading = false;
          throw new Error('Download aborted');
      }

      retries++;
      if (retries >= MAX_RETRIES) {
        isDownloading = false;
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

  if (bufferOffset > 0) {
    await dbPutChunk(chunkIndex, buffer.slice(0, bufferOffset));
    chunkIndex++;
  }

  await dbPutMeta({
    totalSize: received,
    chunkCount: chunkIndex,
    chunkSize: CHUNK_SIZE,
    complete: true,
  });

  isDownloading = false;
  sStatus.textContent = 'Model cached!';
  sBar.style.width = '100%';

  await releaseWakeLock();
}

async function initLLM(meta) {
  if (llm) {
    try {
      if (llm.close) llm.close();
      else if (llm.delete) llm.delete();
    } catch (e) {}
    llm = null;
  }

  sStatus.textContent = 'Initializing AI engine...';
  sDetail.textContent = 'Loading WASM runtime...';
  sBar.style.width = '60%';

  const genai = await FilesetResolver.forGenAiTasks(WASM_PATH);

  sDetail.textContent = 'Loading model into GPU memory...';
  sBar.style.width = '70%';

  let fakeProgress = 70;
  const progressInterval = setInterval(() => {
    fakeProgress += (Math.random() * 2);
    if (fakeProgress > 95) fakeProgress = 95;
    sBar.style.width = fakeProgress + '%';
    sStatus.textContent = `Initializing... ${Math.floor(fakeProgress)}%`;
  }, 200);

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
      const bytes = data instanceof Uint8Array
        ? data
        : new Uint8Array(data instanceof ArrayBuffer ? data : data.buffer || data);
      controller.enqueue(bytes);
      currentChunk++;
    },
  });

  try {
    llm = await LlmInference.createFromOptions(genai, {
        baseOptions: { modelAssetBuffer: stream.getReader() },
        maxTokens: MAX_TOKENS,
        topK: 40,
        temperature: 0.8,
        randomSeed: Math.floor(Math.random() * 100000),
    });
  } finally {
    clearInterval(progressInterval);
  }

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

const backupBtn = $('btn-backup');
if (backupBtn) {
  if ('showSaveFilePicker' in window) {
    backupBtn.style.display = '';
  }
  backupBtn.addEventListener('click', () => exportModelToFile());
}

const altBtns = $('s-alt-btns');
const directDlBtn = $('s-direct-dl');
const loadFileBtn = $('s-load-file');

if (directDlBtn) {
  directDlBtn.addEventListener('click', () => {
    window.open(MODEL_URL, '_blank');
  });
}

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
        if (err.message === 'Download aborted') return;
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

function buildPrompt(userText) {
  const c = cur();
  if (!c) return '';

  const msgs = c.messages;
  const start = Math.max(0, msgs.length - CTX_PAIRS * 2);

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

function esc(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function fmtMsg(text) {
  let s = esc(text);
  s = s.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  s = s.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  return s;
}

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

async function boot() {
  if (!(await gpuOk())) {
    noGpu.classList.add('show');
    setupEl.style.display = 'none';
    return;
  }

  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().catch(() => {});
  }

  loadChats();
  renderSB();
  renderMsgs();
  hdrTitle.textContent = cur()?.name || 'New Chat';
  updateCtx();

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

  sStatus.textContent = 'Preparing to download model...';
  if (altBtns) altBtns.style.display = '';
  try {
    await downloadModel();
    if (altBtns) altBtns.style.display = 'none';
    const meta = await dbGetMeta();
    await initLLM(meta);
  } catch (err) {
    if (err.message === 'Download aborted') return;
    await releaseWakeLock();
    sStatus.textContent = 'Download failed';
    sError.textContent = err.message;
    sBar.style.width = '0%';
  }
}

boot();