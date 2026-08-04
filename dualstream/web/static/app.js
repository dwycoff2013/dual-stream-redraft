const $ = (id) => document.getElementById(id);

function show(data) {
  $("status").textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

async function postJSON(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(JSON.stringify(data));
  }
  return data;
}

async function run() {
  const uiStatus = await fetch("/ui/status").then((r) => r.json());
  $("offline").checked = Boolean(uiStatus.offline_default);

  $("preflight-generate").onclick = async () => {
    try {
      const payload = {
        prompt: $("prompt").value,
        model: $("model").value,
        outdir: $("outdir").value,
        offline: $("offline").checked,
        cache_dir: $("cacheDir").value || null,
      };
      const result = await postJSON("/preflight/generate", payload);
      show(result);
    } catch (err) {
      show(`Preflight error: ${err.message}`);
    }
  };

  $("start-generate").onclick = async () => {
    try {
      const payload = {
        prompt: $("prompt").value,
        model: $("model").value,
        outdir: $("outdir").value,
        offline: $("offline").checked,
        cache_dir: $("cacheDir").value || null,
      };
      const pre = await postJSON("/preflight/generate", payload);
      if (!pre.ok) {
        show({ error: "Preflight failed", preflight: pre });
        return;
      }
      const result = await postJSON("/generate", payload);
      show(result);
    } catch (err) {
      show(`Generate error: ${err.message}`);
    }
  };

  $("preflight-task").onclick = async () => {
    const payload = { task: $("taskPath").value, outdir: $("taskOutdir").value };
    const result = await postJSON("/preflight/arc_solve_task", payload);
    show(result);
  };

  $("start-task").onclick = async () => {
    const payload = { task: $("taskPath").value, outdir: $("taskOutdir").value };
    const pre = await postJSON("/preflight/arc_solve_task", payload);
    if (!pre.ok) {
      show({ error: "Preflight failed", preflight: pre });
      return;
    }
    const result = await postJSON("/arc/solve-task", payload);
    show(result);
  };
}

run().catch((err) => show(`UI boot error: ${err.message}`));
