// API helper for Anomalib test UI
const qs = new URLSearchParams(window.location.search);
export const BACKEND_URL = qs.get("backend") || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  try {
    const res = await fetch(`${BACKEND_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      const err = await res.text();
      return { ok: false, error: err || res.statusText };
    }
    if (res.status === 204) return { ok: true, data: null };
    return { ok: true, data: await res.json() };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

export const api = {
  getProjects: () => request("/api/projects"),
  getPipeline: (id) => request(`/api/projects/${id}/pipeline`),
  enablePipeline: (id) =>
    request(`/api/projects/${id}/pipeline:enable`, { method: "POST" }),
  disablePipeline: (id) =>
    request(`/api/projects/${id}/pipeline:disable`, { method: "POST" }),
  getModels: (id) => request(`/api/projects/${id}/models`),
  patchPipeline: (id, body) =>
    request(`/api/projects/${id}/pipeline`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  getSources: () => request("/api/sources"),
  getSinks: () => request("/api/sinks"),
  createSink: (body) =>
    request("/api/sinks", { method: "POST", body: JSON.stringify(body) }),
  createSource: (body) =>
    request("/api/sources", { method: "POST", body: JSON.stringify(body) }),
  // media
  listMedia: (id) => request(`/api/projects/${id}/images`),
  captureImage: (id, file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BACKEND_URL}/api/projects/${id}/capture`, {
      method: "POST",
      body: form,
    })
      .then(async (res) =>
        res.ok
          ? { ok: true, data: await res.json() }
          : { ok: false, error: await res.text() },
      )
      .catch((e) => ({ ok: false, error: e.message }));
  },
  // jobs
  submitTrainJob: (project_id, model_name = "padim") =>
    request("/api/jobs:train", {
      method: "POST",
      body: JSON.stringify({ project_id, model_name }),
    }),
  // projects
  createProject: (name) =>
    request("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  getProjects: () => request("/api/projects"),
};
