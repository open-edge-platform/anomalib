import { api, BACKEND_URL } from "./api.js";
// runtime fallback in case a cached module misses captureImage
if (!api.captureImage) {
  api.captureImage = (id, file) => {
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
  };
}
import { state, setState, subscribe } from "./state.js";
// ------ Tabs (Inference / Dataset) ------
const tabInf = document.getElementById("tabInference");
const tabData = document.getElementById("tabDataset");
const sidebarInf = document.getElementById("sidebar");
const sidebarData = document.getElementById("sidebarDataset");
function setTab(tab) {
  if (tab === "dataset") {
    sidebarInf.classList.add("hidden");
    sidebarData.classList.remove("hidden");
    localStorage.setItem("anomalib_tab", "dataset");
    // switch video to browser webcam preview instead of WebRTC
    try {
      if (window.webrtcDisconnect) window.webrtcDisconnect();
    } catch {}
    const webVideo = document.getElementById("remoteVideo");
    const dsPrev = document.getElementById("datasetPreview");
    if (webVideo) webVideo.classList.add("hidden");
    if (dsPrev) dsPrev.classList.remove("hidden");
    (async () => {
      try {
        // load projects for dataset tab
        await loadProjectsForDataset();
        const savedProj = localStorage.getItem("anomalib_dataset_project");
        if (
          savedProj &&
          datasetProjectSelect &&
          [...datasetProjectSelect.options].some((o) => o.value === savedProj)
        ) {
          datasetProjectSelect.value = savedProj;
        }
        // populate cameras picker (dataset toolbar)
        const picker = document.getElementById("datasetToolbar");
        const select = document.getElementById("datasetCameras");
        if (picker && select) {
          picker.classList.remove("hidden");
          await navigator.mediaDevices.getUserMedia({ video: true });
          const devs = (await navigator.mediaDevices.enumerateDevices()).filter(
            (d) => d.kind === "videoinput",
          );
          select.innerHTML = "";
          devs.forEach((d, i) => {
            const o = document.createElement("option");
            o.value = d.deviceId || String(i);
            o.textContent = d.label || `Camera ${i + 1}`;
            select.appendChild(o);
          });
        }
        const deviceId = (document.getElementById("datasetCameras") || {})
          .value;
        const constraints = deviceId
          ? { video: { deviceId: { exact: deviceId } }, audio: false }
          : { video: true, audio: false };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        dsPrev.srcObject = stream;
      } catch (e) {
        toast("Webcam error: " + e.message, "error");
      }
    })();
    // show capture bar
    const capBar = document.getElementById("captureBar");
    if (capBar) capBar.classList.remove("hidden");
    const connBar = document.getElementById("connectionBar");
    if (connBar) connBar.classList.add("hidden");
    const autoWrap = document.getElementById("autoConnectWrap");
    if (autoWrap) autoWrap.classList.add("hidden");
    refreshDataset();
  } else {
    sidebarInf.classList.remove("hidden");
    sidebarData.classList.add("hidden");
    localStorage.setItem("anomalib_tab", "inference");
    // switch back to WebRTC
    const webVideo = document.getElementById("remoteVideo");
    const dsPrev = document.getElementById("datasetPreview");
    if (dsPrev && dsPrev.srcObject) {
      dsPrev.srcObject.getTracks().forEach((t) => t.stop());
      dsPrev.srcObject = null;
    }
    if (dsPrev) dsPrev.classList.add("hidden");
    if (webVideo) webVideo.classList.remove("hidden");
    const pickerOld = document.getElementById("cameraPicker");
    if (pickerOld) pickerOld.classList.add("hidden");
    const pickerNew = document.getElementById("datasetToolbar");
    if (pickerNew) pickerNew.classList.add("hidden");
    const capBar = document.getElementById("captureBar");
    if (capBar) capBar.classList.add("hidden");
    const connBar = document.getElementById("connectionBar");
    if (connBar) connBar.classList.remove("hidden");
    const autoWrap = document.getElementById("autoConnectWrap");
    if (autoWrap) autoWrap.classList.remove("hidden");
  }
}
if (tabInf) tabInf.onclick = () => setTab("inference");
if (tabData) tabData.onclick = () => setTab("dataset");
setTab(localStorage.getItem("anomalib_tab") || "inference");

// ------ Dataset logic ------
const datasetGrid = document.getElementById("datasetGrid");
const datasetCount = document.getElementById("datasetCount");
const trainBtn = document.getElementById("trainBtn");
const datasetProjectSelect = document.getElementById("datasetProjectSelect");
const createProjectBtn = document.getElementById("createProjectBtn");
const projectDialog = document.getElementById("projectDialog");
const projectName = document.getElementById("projectName");
const createProjectSubmitBtn = document.getElementById("createProject");
const cancelProject = document.getElementById("cancelProject");

async function refreshDataset() {
  const selectedProjectId = datasetProjectSelect?.value;
  if (!selectedProjectId || !datasetGrid) {
    if (datasetCount) datasetCount.textContent = "0 images";
    if (trainBtn) trainBtn.disabled = true;
    datasetGrid.innerHTML =
      '<div class="muted text-sm p-4">No project selected</div>';
    return 0;
  }

  const res = await api.listMedia(selectedProjectId);
  if (!res.ok) {
    toast(res.error, "error");
    return 0;
  }
  const items = res.data?.media || res.data || [];
  datasetGrid.innerHTML = "";
  const ts = Date.now();
  items.forEach((item) => {
    const img = document.createElement("img");
    img.src = `${BACKEND_URL}/api/projects/${selectedProjectId}/images/${item.id}/full?ts=${ts}`;
    img.alt = item.id;
    img.style.width = "100%";
    img.style.height = "auto";
    img.loading = "lazy";
    datasetGrid.appendChild(img);
  });
  const n = items.length;
  if (datasetCount)
    datasetCount.textContent = `${n} image${n === 1 ? "" : "s"}`;
  if (trainBtn) trainBtn.disabled = n < 20;
  return n;
}

// Project creation and selection logic
async function loadProjectsForDataset() {
  if (!datasetProjectSelect) return;
  const res = await api.getProjects();
  if (!res.ok) {
    toast(res.error || "Failed to load projects", "error");
    return;
  }

  const list = Array.isArray(res.data?.projects)
    ? res.data.projects
    : Array.isArray(res.data)
      ? res.data
      : [];
  datasetProjectSelect.innerHTML =
    '<option value="">Select project...</option>';
  list.forEach((project) => {
    if (!project?.id) return;
    const option = document.createElement("option");
    option.value = project.id;
    option.textContent = project.name || project.id;
    datasetProjectSelect.appendChild(option);
  });

  // Try to keep current selection if possible
  const current = localStorage.getItem("anomalib_dataset_project");
  if (
    current &&
    [...datasetProjectSelect.options].some((o) => o.value === current)
  ) {
    datasetProjectSelect.value = current;
  }
}

// Project creation dialog
if (createProjectBtn) {
  createProjectBtn.onclick = () => {
    projectName.value = "";
    projectDialog.showModal();
  };
}

if (cancelProject) {
  cancelProject.onclick = () => {
    projectDialog.close();
  };
}

if (createProjectSubmitBtn) {
  createProjectSubmitBtn.onclick = async (e) => {
    e.preventDefault();
    const name = projectName.value.trim();
    if (!name) {
      toast("Please enter a project name", "error");
      return;
    }

    createProjectSubmitBtn.disabled = true;
    const res = await api.createProject(name);
    createProjectSubmitBtn.disabled = false;

    if (res.ok) {
      toast("Project created successfully", "success");
      projectDialog.close();
      await loadProjectsForDataset();
      const newId =
        res.data?.id || res.data?.project?.id || res.data?.uuid || "";
      if (newId) {
        datasetProjectSelect.value = newId;
        localStorage.setItem("anomalib_dataset_project", newId);
      }
      refreshDataset();
    } else {
      toast(res.error, "error");
    }
  };
}

// Project selection change
if (datasetProjectSelect) {
  datasetProjectSelect.onchange = () => {
    const val = datasetProjectSelect.value || "";
    if (val) localStorage.setItem("anomalib_dataset_project", val);
    refreshDataset();
  };
  // Load projects list on focus/click to ensure it's fresh
  const ensureLoaded = async () => {
    if (
      !datasetProjectSelect.options ||
      datasetProjectSelect.options.length <= 1
    ) {
      await loadProjectsForDataset();
    }
  };
  datasetProjectSelect.addEventListener("focus", ensureLoaded, {
    passive: true,
  });
  datasetProjectSelect.addEventListener("click", ensureLoaded, {
    passive: true,
  });
}

// capture button: snapshot current frame from video and upload
const captureBtn = document.getElementById("captureBtn");
if (captureBtn)
  captureBtn.onclick = async () => {
    try {
      // prefer datasetPreview when in dataset tab; fallback to remoteVideo
      const dsPrev = document.getElementById("datasetPreview");
      const inDataset =
        sidebarData && !sidebarData.classList.contains("hidden");
      const video =
        inDataset && dsPrev && dsPrev.srcObject
          ? dsPrev
          : document.getElementById("remoteVideo");
      if (!video || !video.videoWidth) {
        toast("No video to capture", "error");
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.92),
      );
      const file = new File([blob], `capture_${Date.now()}.jpg`, {
        type: "image/jpeg",
      });
      const selectedProjectId = datasetProjectSelect?.value;
      if (!selectedProjectId) {
        toast("Please select a project first", "error");
        return;
      }
      const r = await api.captureImage(selectedProjectId, file);
      toast(r.ok ? "Captured" : r.error, r.ok ? "success" : "error");
      if (r.ok) {
        refreshDataset();
        setTimeout(refreshDataset, 800);
      }
    } catch (e) {
      toast(e.message, "error");
    }
  };

// camera switch
const cameraSelect = document.getElementById("datasetCameras");
if (cameraSelect)
  cameraSelect.onchange = async () => {
    try {
      const dsPrev = document.getElementById("datasetPreview");
      if (dsPrev && dsPrev.srcObject) {
        dsPrev.srcObject.getTracks().forEach((t) => t.stop());
      }
      const deviceId = cameraSelect.value;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId } },
        audio: false,
      });
      dsPrev.srcObject = stream;
    } catch (e) {
      toast("Switch camera error: " + e.message, "error");
    }
  };

// train submission
if (trainBtn)
  trainBtn.onclick = async () => {
    const selectedProjectId = datasetProjectSelect?.value;
    if (!selectedProjectId) {
      toast("Please select a project first", "error");
      return;
    }
    trainBtn.disabled = true;
    const modelSel = document.getElementById("trainModelSelect");
    const modelName = (modelSel && modelSel.value) || "padim";
    const res = await api.submitTrainJob(selectedProjectId, modelName);
    toast(
      res.ok ? "Training job submitted" : res.error,
      res.ok ? "success" : "error",
    );
    // re-enable after toast unless count is still <20
    const cntText = datasetCount?.textContent || "0";
    const n = parseInt(cntText, 10) || 0;
    trainBtn.disabled = n < 20;
  };

// ------ toast -------
const toast = (msg, type = "info") => {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.className = `toast show toast-${type}`;
  window.clearTimeout(el._t);
  el._t = window.setTimeout(() => {
    el.textContent = "";
    el.className = "toast hidden";
  }, 3000);
};

// ------ theme toggle and backend label ------
document.getElementById("backendLabel").textContent = BACKEND_URL;
document.getElementById("themeToggle").onclick = () => {
  const root = document.documentElement;
  const isDark = root.classList.toggle("dark");
  document.getElementById("themeToggle").textContent = isDark
    ? "Light"
    : "Dark";
};
// sidebar toggle
const layoutEl = document.getElementById("layout");
const sidebarToggle = document.getElementById("sidebarToggle");
const midSidebarToggle = document.getElementById("midSidebarToggle");
// restore saved sidebar state
if (localStorage.getItem("anomalib_sidebar_collapsed") === "1") {
  layoutEl.classList.add("collapsed");
  if (sidebarToggle) sidebarToggle.textContent = "Show Config";
  if (midSidebarToggle) midSidebarToggle.textContent = "‹";
}
function toggleSidebar() {
  layoutEl.classList.toggle("collapsed");
  const collapsed = layoutEl.classList.contains("collapsed");
  if (sidebarToggle)
    sidebarToggle.textContent = collapsed ? "Show Config" : "Config";
  if (midSidebarToggle) midSidebarToggle.textContent = collapsed ? "‹" : "›";
  localStorage.setItem("anomalib_sidebar_collapsed", collapsed ? "1" : "0");
}
if (sidebarToggle) sidebarToggle.onclick = toggleSidebar;
if (midSidebarToggle) midSidebarToggle.onclick = toggleSidebar;

// ------ Sidebar drag and drop order ------
const sidebar = document.getElementById("sidebar");
const CARD_SELECTOR = ".sidebar-card";
const ORDER_KEY = "anomalib_sidebar_order_v1";

function getCurrentOrder() {
  return Array.from(sidebar.querySelectorAll(CARD_SELECTOR)).map((el) =>
    el.getAttribute("data-key"),
  );
}

function applyOrder(order) {
  if (!order || !order.length) return;
  const map = new Map();
  Array.from(sidebar.querySelectorAll(CARD_SELECTOR)).forEach((el) =>
    map.set(el.getAttribute("data-key"), el),
  );
  order.forEach((key) => {
    const el = map.get(key);
    if (el) sidebar.appendChild(el);
  });
}

function saveOrder() {
  const order = getCurrentOrder();
  localStorage.setItem(ORDER_KEY, JSON.stringify(order));
}

function loadOrder() {
  try {
    const raw = localStorage.getItem(ORDER_KEY);
    if (!raw) return;
    const order = JSON.parse(raw);
    applyOrder(order);
  } catch {}
}

let dragSrc = null;
sidebar.addEventListener("dragstart", (e) => {
  const card = e.target.closest(CARD_SELECTOR);
  if (!card) return;
  dragSrc = card;
  card.classList.add("dnd-ghost");
  e.dataTransfer.effectAllowed = "move";
  e.dataTransfer.setData("text/plain", card.getAttribute("data-key"));
});
sidebar.addEventListener("dragend", (e) => {
  const card = e.target.closest(CARD_SELECTOR);
  if (card) card.classList.remove("dnd-ghost");
  dragSrc = null;
});
sidebar.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = "move";
  const targetCard = e.target.closest(CARD_SELECTOR);
  if (!targetCard || targetCard === dragSrc) return;
  const rect = targetCard.getBoundingClientRect();
  const before = (e.clientY - rect.top) / rect.height < 0.5;
  if (before) sidebar.insertBefore(dragSrc, targetCard);
  else sidebar.insertBefore(dragSrc, targetCard.nextSibling);
});
sidebar.addEventListener("drop", (e) => {
  e.preventDefault();
  saveOrder();
});

// Apply saved order at startup
loadOrder();

// ------ Projects ------
let allProjects = [];
const projectSelect = document.getElementById("projectSelect");
const projectIdInput = document.getElementById("projectIdInput");

async function loadProjects(selectLatest = false) {
  const res = await api.getProjects();
  if (!res.ok) return toast(res.error);
  const list = res.data.projects || [];
  allProjects = list;
  projectSelect.innerHTML = list.length ? "" : "<option>(none)</option>";
  list.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name || p.id;
    projectSelect.appendChild(opt);
  });
  if (selectLatest && list.length) {
    const latest = list[list.length - 1];
    projectSelect.value = latest.id;
    setState({ projectId: latest.id });
  }
}

document.getElementById("refreshProjectsBtn").onclick = () =>
  loadProjects(true);
projectSelect.onchange = () => {
  const id = projectSelect.value;
  setState({ projectId: id });
  // also trigger a deferred refresh to ensure state has propagated
  setTimeout(() => {
    try {
      refreshModels();
    } catch {}
    try {
      refreshSources();
    } catch {}
    try {
      refreshSinks();
    } catch {}
    try {
      loadPipeline();
    } catch {}
  }, 50);
};
projectIdInput.onchange = () => {
  const id = projectIdInput.value.trim() || null;
  setState({ projectId: id });
  setTimeout(() => {
    try {
      refreshModels();
    } catch {}
    try {
      refreshSources();
    } catch {}
    try {
      refreshSinks();
    } catch {}
    try {
      loadPipeline();
    } catch {}
  }, 50);
};

// reflect projectId in input field and select
subscribe("projectId", (id) => {
  if (projectIdInput) projectIdInput.value = id || "";
  if (projectSelect && id) projectSelect.value = id;
  // if Dataset tab is active, refresh its content on project switch
  if (
    document.getElementById("sidebarDataset") &&
    !document.getElementById("sidebarDataset").classList.contains("hidden")
  ) {
    refreshDataset();
  }
  if (datasetProjectId) datasetProjectId.textContent = id || "";
  // refresh dependent lists for inference tab
  try {
    refreshModels();
  } catch {}
  try {
    refreshSources();
  } catch {}
  try {
    refreshSinks();
  } catch {}
  try {
    loadPipeline();
  } catch {}
});

// ------ Pipeline ------
let pipelinePoll = null;
subscribe("pipeline", (p) => {
  const span = document.getElementById("pipelineStatus");
  span.textContent = "";
  span.className = "text-sm muted";
  if (p) {
    const chip = document.createElement("span");
    chip.className =
      "chip " +
      (p.status === "RUNNING"
        ? "chip-ok"
        : p.status === "IDLE"
          ? "chip-muted"
          : "chip-error");
    chip.textContent = p.status;
    span.appendChild(chip);
  }
  // Keep enable/disable buttons always active per requirement
  const en = document.getElementById("enableBtn");
  const dis = document.getElementById("disableBtn");
  if (en) en.disabled = false;
  if (dis) dis.disabled = false;

  // auto-refresh while running
  if (p && p.status === "RUNNING") {
    if (!pipelinePoll) pipelinePoll = window.setInterval(loadPipeline, 3000);
  } else {
    if (pipelinePoll) {
      window.clearInterval(pipelinePoll);
      pipelinePoll = null;
    }
  }
});

async function loadPipeline() {
  if (!state.projectId) return;
  const res = await api.getPipeline(state.projectId);
  if (!res.ok) return toast(res.error);
  setState({ pipeline: res.data });
}

document.getElementById("enableBtn").onclick = async () => {
  if (!state.projectId) return;
  toast("Enabling pipeline...", "info");
  const r = await api.enablePipeline(state.projectId);
  toast(r.ok ? "Enabled" : r.error, r.ok ? "success" : "error");
  loadPipeline();
};
document.getElementById("disableBtn").onclick = async () => {
  if (!state.projectId) return;
  toast("Disabling pipeline...", "info");
  const r = await api.disablePipeline(state.projectId);
  toast(r.ok ? "Disabled" : r.error, r.ok ? "success" : "error");
  loadPipeline();
};

// ------ Models ------
async function refreshModels() {
  if (!state.projectId) return;
  const res = await api.getModels(state.projectId);
  if (!res.ok) return toast(res.error);
  setState({ models: res.data.models || res.data });
}

let modelsAll = [];
subscribe("models", (list) => {
  modelsAll = list || [];
  const sel = document.getElementById("modelsSelect");
  const empty = document.getElementById("modelsEmpty");
  sel.innerHTML = modelsAll.length ? "" : "<option>(none)</option>";
  empty.classList.toggle("hidden", !!modelsAll.length);
  modelsAll.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    sel.appendChild(opt);
  });
});

const modelsSearch = document.getElementById("modelsSearch");
modelsSearch.oninput = () => {
  const q = modelsSearch.value.toLowerCase();
  const sel = document.getElementById("modelsSelect");
  sel.innerHTML = "";
  modelsAll
    .filter((m) => (m.name || "").toLowerCase().includes(q))
    .forEach((m) => {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.name;
      sel.appendChild(o);
    });
};

document.getElementById("refreshModelsBtn").onclick = refreshModels;
document.getElementById("refreshSourcesBtn").onclick = refreshSources;
document.getElementById("setModelBtn").onclick = async () => {
  const id = document.getElementById("modelsSelect").value;
  if (!id) return;
  const r = await api.patchPipeline(state.projectId, { model_id: id });
  if (r.ok) {
    document.getElementById("modelsUpdated").textContent = "Just now ✓";
  }
  toast(r.ok ? "Model set" : r.error, r.ok ? "success" : "error");
  loadPipeline();
};

// ------ Sources ------
async function refreshSources() {
  const r = await api.getSources();
  if (!r.ok) return toast(r.error);
  setState({ sources: r.data });
}
let sourcesAll = [];
subscribe("sources", (list) => {
  sourcesAll = list || [];
  const sel = document.getElementById("sourcesSelect");
  const empty = document.getElementById("sourcesEmpty");
  sel.innerHTML = sourcesAll.length ? "" : "<option>(none)</option>";
  empty.classList.toggle("hidden", !!sourcesAll.length);
  sourcesAll.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.id;
    o.textContent = `${s.name} (${s.source_type})`;
    sel.appendChild(o);
  });
});

const sourcesSearch = document.getElementById("sourcesSearch");
sourcesSearch.oninput = () => {
  const q = sourcesSearch.value.toLowerCase();
  const sel = document.getElementById("sourcesSelect");
  sel.innerHTML = "";
  sourcesAll
    .filter((s) => `${s.name} ${s.source_type}`.toLowerCase().includes(q))
    .forEach((s) => {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = `${s.name} (${s.source_type})`;
      sel.appendChild(o);
    });
};

document.getElementById("setSourceBtn").onclick = async () => {
  const id = document.getElementById("sourcesSelect").value;
  if (!id) return;
  const r = await api.patchPipeline(state.projectId, { source_id: id });
  if (r.ok) {
    document.getElementById("sourcesUpdated").textContent = "Just now ✓";
  }
  toast(r.ok ? "Source set" : r.error, r.ok ? "success" : "error");
  loadPipeline();
};

// ------ Sinks ------
async function refreshSinks() {
  const r = await api.getSinks();
  if (!r.ok) return toast(r.error);
  setState({ sinks: r.data });
}
let sinksAll = [];
subscribe("sinks", (list) => {
  sinksAll = list || [];
  const sel = document.getElementById("sinksSelect");
  const empty = document.getElementById("sinksEmpty");
  sel.innerHTML = sinksAll.length ? "" : "<option>(none)</option>";
  empty.classList.toggle("hidden", !!sinksAll.length);
  sinksAll.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.id;
    o.textContent = `${s.name} (${s.sink_type})`;
    sel.appendChild(o);
  });
});

document.getElementById("refreshSinksBtn").onclick = refreshSinks;
document.getElementById("setSinkBtn").onclick = async () => {
  const id = document.getElementById("sinksSelect").value;
  if (!id) return;
  const r = await api.patchPipeline(state.projectId, { sink_id: id });
  if (r.ok) {
    document.getElementById("sinksUpdated").textContent = "Just now ✓";
  }
  toast(r.ok ? "Sink set" : r.error, r.ok ? "success" : "error");
  loadPipeline();
};

document.getElementById("useDefaultSinkBtn").onclick = async () => {
  if (!state.projectId) {
    toast("Select a project first", "error");
    return;
  }
  const btn = document.getElementById("useDefaultSinkBtn");
  if (btn) btn.disabled = true;
  try {
    const DEFAULT_SINK_PATH = "./sink";
    const DEFAULT_SINK_NAME = "Default Folder Sink";
    const normalizePath = (p) =>
      (p || "").replace(/[/\\]+$/, "").replace(/\\/g, "/");
    const targetPath = normalizePath(DEFAULT_SINK_PATH);
    const findDefaultFolderSink = () =>
      sinksAll.find(
        (s) =>
          s.sink_type === "folder" &&
          normalizePath(s.folder_path || "") === targetPath,
      );

    let sinkId = findDefaultFolderSink()?.id;
    if (!sinkId) {
      const res = await api.createSink({
        sink_type: "folder",
        name: DEFAULT_SINK_NAME,
        folder_path: DEFAULT_SINK_PATH,
        output_formats: [
          "image_original",
          "image_with_predictions",
          "predictions",
        ],
      });
      if (!res.ok) {
        throw new Error(res.error || "Failed to create default sink");
      }
      sinkId = res.data?.id || res.data?.sink?.id;
      await refreshSinks();
      const created = findDefaultFolderSink();
      if (created?.id) sinkId = created.id;
    }
    if (!sinkId) {
      throw new Error("Default sink id missing");
    }
    const patchRes = await api.patchPipeline(state.projectId, {
      sink_id: sinkId,
    });
    if (patchRes.ok) {
      document.getElementById("sinksUpdated").textContent = "Just now ✓";
    }
    toast(
      patchRes.ok ? `Default sink set to ${DEFAULT_SINK_PATH}` : patchRes.error,
      patchRes.ok ? "success" : "error",
    );
    loadPipeline();
  } catch (error) {
    toast(error.message || "Failed to set default sink", "error");
  } finally {
    if (btn) btn.disabled = false;
  }
};

// ------ Source dialog ------
const dlg = document.getElementById("sourceDialog");
const typeSel = document.getElementById("srcTypeSel");
const camSec = document.getElementById("camSection");
const pathSec = document.getElementById("pathSection");
typeSel.onchange = () => {
  const cam = typeSel.value === "webcam";
  camSec.classList.toggle("hidden", !cam);
  pathSec.classList.toggle("hidden", cam);
};

document.getElementById("addSourceBtn").onclick = () => dlg.showModal();
document.getElementById("cancelSrcBtn").onclick = () => dlg.close();

document.getElementById("detectBtn").onclick = async () => {
  const status = document.getElementById("srcDialogStatus");
  try {
    status.textContent = "Requesting camera access...";
    await navigator.mediaDevices.getUserMedia({ video: true });
    status.textContent = "Enumerating cameras...";
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(
      (d) => d.kind === "videoinput",
    );
    const sel = document.getElementById("camSelect");
    sel.innerHTML = '<option value="">Select camera...</option>';

    if (devs.length === 0) {
      status.textContent = "No cameras found";
      return;
    }

    devs.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = String(i); // index sent to backend as integer device_id
      o.dataset.deviceId = d.deviceId || "";
      o.textContent = d.label || `Camera ${i + 1}`;
      sel.appendChild(o);
    });
    status.textContent = `${devs.length} camera(s) found - select one to preview`;
  } catch (e) {
    console.error("Camera detection error:", e);
    status.textContent = `Camera detection failed: ${e.message}`;
  }
};

// live camera preview lifecycle
let previewStream = null;
document.getElementById("camSelect").onchange = async () => {
  const selEl = document.getElementById("camSelect");
  const opt = selEl && selEl.selectedOptions && selEl.selectedOptions[0];
  const deviceId = opt ? opt.dataset.deviceId || "" : "";
  const pv = document.getElementById("camPreview");
  const status = document.getElementById("srcDialogStatus");

  if (previewStream) {
    previewStream.getTracks().forEach((t) => t.stop());
    previewStream = null;
  }

  if (!deviceId) {
    pv.classList.add("hidden");
    pv.srcObject = null;
    status.textContent = "Select a camera to preview";
    return;
  }

  try {
    status.textContent = "Starting camera preview...";
    const constraints = deviceId
      ? { video: { deviceId: { exact: deviceId } }, audio: false }
      : { video: true, audio: false };

    previewStream = await navigator.mediaDevices.getUserMedia(constraints);
    pv.srcObject = previewStream;
    pv.classList.remove("hidden");
    await pv.play();
    status.textContent = "Camera preview active";
  } catch (e) {
    console.error("Camera preview error:", e);
    status.textContent = `Preview failed: ${e.message}`;
    pv.classList.add("hidden");
    pv.srcObject = null;
  }
};
dlg.addEventListener("close", () => {
  if (previewStream) {
    previewStream.getTracks().forEach((t) => t.stop());
    previewStream = null;
  }
  const pv = document.getElementById("camPreview");
  pv.srcObject = null;
  pv.classList.add("hidden");
});

document.getElementById("createSrcBtn").onclick = async (e) => {
  e.preventDefault();
  const name = document.getElementById("srcName").value.trim();
  if (!name) {
    document.getElementById("srcDialogStatus").textContent = "Enter a name";
    return;
  }
  const body = { name };
  if (typeSel.value === "webcam") {
    const selEl = document.getElementById("camSelect");
    const value = selEl && selEl.value;
    if (!value) {
      document.getElementById("srcDialogStatus").textContent =
        "Select a camera";
      return;
    }
    body.source_type = "webcam";
    body.device_id = parseInt(value, 10); // backend expects integer index
  } else if (typeSel.value === "video_file") {
    body.source_type = "video_file";
    body.video_path = document.getElementById("srcPath").value.trim();
  } else {
    body.source_type = "images_folder";
    body.images_folder_path = document.getElementById("srcPath").value.trim();
    body.ignore_existing_images = true;
  }
  const r = await api.createSource(body);
  document.getElementById("srcDialogStatus").textContent = r.ok
    ? "Source created"
    : r.error;
  toast(r.ok ? "Source created" : r.error, r.ok ? "success" : "error");
  if (r.ok) {
    dlg.close();
    refreshSources();
  }
};

// ------ initial load ------
(async function init() {
  // default theme from prefers-color-scheme
  if (
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    const user = localStorage.getItem("anomalib_theme");
    const wantDark = user ? user === "dark" : true;
    if (wantDark) {
      document.documentElement.classList.add("dark");
      document.getElementById("themeToggle").textContent = "Light";
    }
  }
  document.getElementById("themeToggle").addEventListener("click", () => {
    const isDark = document.documentElement.classList.contains("dark");
    localStorage.setItem("anomalib_theme", isDark ? "light" : "dark");
  });
  await loadProjects(true);
  await refreshSources();
  await refreshSinks();
  await refreshModels();
  await loadPipeline();
  // final safeguard to ensure buttons are active
  const enBtn = document.getElementById("enableBtn");
  const disBtn = document.getElementById("disableBtn");
  if (enBtn) enBtn.disabled = false;
  if (disBtn) disBtn.disabled = false;
})();
