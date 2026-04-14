import { api } from '/static/api-client.js';

const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
const state = { tab: 'generate', jobs: [], selectedJobId: null, poller: null };

const el = {
  tabs: Array.from(document.querySelectorAll('.tab')),
  sections: Array.from(document.querySelectorAll('[data-section]')),
  form: document.getElementById('job-form'),
  submitBtn: document.getElementById('submit-btn'),
  cancelBtn: document.getElementById('cancel-btn'),
  refreshBtn: document.getElementById('refresh-btn'),
  jobsList: document.getElementById('jobs-list'),
  selectedJob: document.getElementById('selected-job'),
  jobMeta: document.getElementById('job-meta'),
  status: document.getElementById('status-banner'),
  error: document.getElementById('error-banner'),
  result: document.getElementById('result-json'),
  errorJson: document.getElementById('error-json'),
  artifacts: document.getElementById('artifacts-body'),
};

function showError(message) {
  el.error.textContent = message;
  el.error.classList.remove('hidden');
}
function clearError() { el.error.classList.add('hidden'); el.error.textContent = ''; }
function setStatus(message) { el.status.textContent = message; }

function switchTab(tab) {
  state.tab = tab;
  el.tabs.forEach((btn) => btn.classList.toggle('active', btn.dataset.tab === tab));
  el.sections.forEach((section) => section.classList.toggle('hidden', section.dataset.section !== tab));
}

function activeJob() {
  return state.jobs.find((j) => j.id === state.selectedJobId) || null;
}

function renderJobs() {
  el.jobsList.innerHTML = '';
  if (!state.jobs.length) {
    el.jobsList.innerHTML = '<li class="subtle">No jobs yet.</li>';
    return;
  }
  state.jobs.forEach((job) => {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = `${job.kind} • ${job.status} • ${job.id.slice(0, 8)}`;
    btn.setAttribute('aria-pressed', String(job.id === state.selectedJobId));
    btn.addEventListener('click', () => { state.selectedJobId = job.id; renderSelectedJob(); });
    li.appendChild(btn);
    el.jobsList.appendChild(li);
  });
}

async function renderSelectedJob() {
  const job = activeJob();
  if (!job) {
    el.selectedJob.textContent = 'None';
    el.jobMeta.textContent = 'No job selected.';
    el.result.textContent = '{}';
    el.errorJson.textContent = 'No errors.';
    el.cancelBtn.disabled = true;
    renderArtifacts([]);
    return;
  }

  el.selectedJob.textContent = job.id;
  el.jobMeta.textContent = `${job.kind} • ${job.status} • progress ${Math.round((job.progress || 0) * 100)}% • ${job.message || ''}`;
  el.result.textContent = JSON.stringify(job.result || {}, null, 2);
  el.errorJson.textContent = job.error || JSON.stringify(job.logs || [], null, 2) || 'No errors.';
  el.cancelBtn.disabled = TERMINAL.has(job.status);

  try {
    const artifacts = await api.getArtifacts(job.id);
    renderArtifacts(Array.isArray(artifacts) ? artifacts : []);
  } catch (error) {
    renderArtifacts([]);
    showError(`Artifacts unavailable: ${error.message}`);
  }
}

function renderArtifacts(artifacts) {
  el.artifacts.innerHTML = '';
  if (!artifacts.length || !state.selectedJobId) {
    el.artifacts.innerHTML = '<tr><td colspan="4" class="subtle">No artifacts.</td></tr>';
    return;
  }
  artifacts.forEach((artifact) => {
    const row = document.createElement('tr');
    const url = artifact.url || `/artifacts/${encodeURIComponent(state.selectedJobId)}/file/${artifact.relative_path}`;
    row.innerHTML = `
      <td>${artifact.name || ''}</td>
      <td>${artifact.relative_path || ''}</td>
      <td>${artifact.size ?? ''}</td>
      <td><a href="${url}" target="_blank" rel="noopener">Open</a></td>
    `;
    el.artifacts.appendChild(row);
  });
}

async function loadJobs({ announce = false } = {}) {
  try {
    const jobs = await api.listJobs();
    state.jobs = Array.isArray(jobs) ? jobs : [];
    if (!state.selectedJobId && state.jobs.length) {
      state.selectedJobId = state.jobs[0].id;
    }
    renderJobs();
    await renderSelectedJob();
    managePolling();
    if (announce) setStatus(`Loaded ${state.jobs.length} jobs.`);
  } catch (error) {
    showError(`Failed to load jobs: ${error.message}`);
    setStatus('Unable to refresh jobs. Backend may still be starting.');
  }
}

function managePolling() {
  const shouldPoll = state.jobs.some((job) => !TERMINAL.has(job.status));
  if (shouldPoll && !state.poller) {
    state.poller = setInterval(() => loadJobs(), 2000);
  }
  if (!shouldPoll && state.poller) {
    clearInterval(state.poller);
    state.poller = null;
  }
}

function payloadFromForm() {
  const fd = new FormData(el.form);
  const outdir = String(fd.get('outdir') || '').trim();
  if (state.tab === 'generate') {
    return {
      route: '/generate',
      payload: {
        outdir,
        prompt: String(fd.get('prompt') || '').trim(),
        model: String(fd.get('model') || 'gpt2').trim(),
        max_new_tokens: Number(fd.get('max_new_tokens') || 128),
      },
    };
  }
  if (state.tab === 'solve-task') {
    return { route: '/arc/solve-task', payload: { outdir, task: String(fd.get('task') || '').trim() } };
  }
  if (state.tab === 'solve-dataset') {
    return { route: '/arc/solve-dataset', payload: { outdir, tasks_dir: String(fd.get('tasks_dir') || '').trim() } };
  }
  return {
    route: '/arc/kaggle-submit',
    payload: { tasks_dir: String(fd.get('kaggle_tasks_dir') || '').trim(), output: String(fd.get('output') || '').trim() },
  };
}

async function submitJob(event) {
  event.preventDefault();
  clearError();
  el.submitBtn.disabled = true;
  try {
    const { route, payload } = payloadFromForm();
    const response = await api.submit(route, payload);
    const jobId = response?.job_id;
    if (!jobId) throw new Error('Job started but no job_id was returned.');
    state.selectedJobId = jobId;
    setStatus(`Started ${state.tab} job: ${jobId}`);
    await loadJobs();
  } catch (error) {
    showError(`Failed to start job: ${error.message}`);
  } finally {
    el.submitBtn.disabled = false;
  }
}

async function cancelSelectedJob() {
  clearError();
  const jobId = state.selectedJobId;
  if (!jobId) {
    showError('Select a job first.');
    return;
  }
  try {
    await api.cancelJob(jobId);
    setStatus(`Cancellation requested for ${jobId}`);
    await loadJobs();
  } catch (error) {
    showError(`Cancel failed: ${error.message}`);
  }
}

el.tabs.forEach((btn) => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));
el.form.addEventListener('submit', submitJob);
el.cancelBtn.addEventListener('click', cancelSelectedJob);
el.refreshBtn.addEventListener('click', () => loadJobs({ announce: true }));

switchTab('generate');
api.health().then(() => loadJobs({ announce: true })).catch(() => {
  setStatus('Backend health check failed. Try refresh in a moment.');
  loadJobs();
});
