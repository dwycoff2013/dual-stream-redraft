async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
  } catch (error) {
    throw new Error(`Network error while calling ${path}: ${error.message}`);
  }

  let data = null;
  const text = await response.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      if (!response.ok) {
        throw new Error(`Request failed (${response.status}) and returned invalid JSON.`);
      }
      return text;
    }
  }

  if (!response.ok) {
    const detail = data?.detail || `HTTP ${response.status}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

export const api = {
  health: () => request('/health'),
  listJobs: () => request('/jobs'),
  getJob: (jobId) => request(`/jobs/${encodeURIComponent(jobId)}`),
  cancelJob: (jobId) => request(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
  getArtifacts: (jobId) => request(`/artifacts/${encodeURIComponent(jobId)}`),
  listScripts: () => request('/scripts'),
  submit: (route, payload) => request(route, { method: 'POST', body: JSON.stringify(payload) }),
};
