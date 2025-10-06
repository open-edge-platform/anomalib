// Pipeline Control JavaScript
class PipelineController {
  constructor() {
    this.currentPipeline = null;
    this.currentSource = null;
    this.currentModel = null;
    this.currentSink = null;
    this.sources = [];
    this.models = [];
    this.sinks = [];

    console.log("PipelineController constructor called");
    console.log("DOM ready state:", document.readyState);
    console.log(
      "projectId element exists:",
      !!document.getElementById("projectId"),
    );

    this.initializeEventListeners();
    this.loadInitialData();
  }

  initializeEventListeners() {
    // Pipeline controls
    document.getElementById("loadPipelineBtn").addEventListener("click", () => {
      const projectIdInput = document.getElementById("projectId");
      if (!projectIdInput) {
        console.error("Project ID input element not found!");
        this.updateStatus(
          "pipelineStatus",
          "Project ID input not found",
          "error",
        );
        return;
      }
      this.projectId = projectIdInput.value;
      if (this.projectId) {
        this.loadPipeline();
      } else {
        this.updateStatus(
          "pipelineStatus",
          "Please enter a valid project ID",
          "error",
        );
      }
    });
    document
      .getElementById("enablePipelineBtn")
      .addEventListener("click", () => this.enablePipeline());
    document
      .getElementById("disablePipelineBtn")
      .addEventListener("click", () => this.disablePipeline());

    // Source controls
    document
      .getElementById("updateSourceBtn")
      .addEventListener("click", () => this.updateSource());
    document
      .getElementById("addSourceBtn")
      .addEventListener("click", () => this.showAddSourceModal());
    document
      .getElementById("sourceSelect")
      .addEventListener("change", () => this.enableUpdateSourceBtn());

    // Model controls
    document
      .getElementById("updateModelBtn")
      .addEventListener("click", () => this.updateModel());

    // Sink controls
    document
      .getElementById("updateSinkBtn")
      .addEventListener("click", () => this.updateSink());

    // Project ID change
    const projectIdInput = document.getElementById("projectId");
    if (projectIdInput) {
      projectIdInput.addEventListener("change", (e) => {
        this.projectId = e.target.value;
        if (this.projectId) {
          this.loadPipeline();
        }
      });
    } else {
      console.error("Project ID input element not found for event listener!");
    }

    // Modal controls
    this.initializeModalEventListeners();
  }

  initializeModalEventListeners() {
    // Modal open/close
    document
      .getElementById("closeModal")
      .addEventListener("click", () => this.hideAddSourceModal());
    document
      .getElementById("cancelSourceBtn")
      .addEventListener("click", () => this.hideAddSourceModal());

    // Modal source type change
    document
      .getElementById("modalSourceType")
      .addEventListener("change", (e) =>
        this.toggleModalSourceType(e.target.value),
      );

    // Modal camera controls
    document
      .getElementById("modalDetectCamerasBtn")
      .addEventListener("click", () => this.modalDetectCameras());
    document
      .getElementById("modalCameraSelect")
      .addEventListener("change", (e) =>
        this.modalPreviewCamera(e.target.value),
      );
    document
      .getElementById("modalStopPreviewBtn")
      .addEventListener("click", () => this.modalStopCameraPreview());

    // Modal file path input
    document
      .getElementById("modalSourcePath")
      .addEventListener("input", () => this.enableModalCreateBtn());
    document
      .getElementById("modalSourceName")
      .addEventListener("input", () => this.enableModalCreateBtn());

    // Create source
    document
      .getElementById("createSourceBtn")
      .addEventListener("click", () => this.createSource());
  }

  async loadInitialData() {
    await this.loadLatestProject();
    await this.loadSources();
    await this.loadSinks();
    // Load models and pipeline only if we have a valid project ID
    if (this.projectId) {
      await this.loadModels();
      await this.loadPipeline();
    }
  }

  async loadLatestProject() {
    try {
      // Step 1: GET list of projects
      const resp = await fetch(`${window.BACKEND_URL}/api/projects`);
      if (!resp.ok) {
        console.warn("Failed to load projects:", resp.status);
        return;
      }
      const data = await resp.json();

      if (data.projects && data.projects.length > 0) {
        // Step 2: Choose the latest project (last in the list)
        const latest = data.projects[data.projects.length - 1];

        // Step 3: Get the id of that project
        const projectId = latest.id;

        // Step 4: Put that id in the text box of the pipeline control - project id
        const input = document.getElementById("projectId");
        if (!input) {
          console.error("Project ID input element not found!");
          return;
        }
        input.value = projectId;

        // Step 5: Set this.projectId for use in loadPipeline
        this.projectId = projectId;

        console.log("Loaded latest project:", projectId, latest.name);
      } else {
        console.warn("No projects found");
      }
    } catch (e) {
      console.error("Error loading latest project:", e);
    }
  }

  async loadSources() {
    try {
      const response = await fetch(`${window.BACKEND_URL}/api/sources`);
      if (!response.ok)
        throw new Error(`Failed to load sources: ${response.status}`);

      this.sources = await response.json();
      this.populateSourceSelect();
    } catch (error) {
      console.error("Error loading sources:", error);
      this.updateStatus(
        "sourceStatus",
        `Error loading sources: ${error.message}`,
        "error",
      );
    }
  }

  populateSourceSelect() {
    const select = document.getElementById("sourceSelect");
    select.innerHTML = '<option value="">Select a source...</option>';

    this.sources.forEach((source) => {
      const option = document.createElement("option");
      option.value = source.id;
      option.textContent = `${source.name} (${source.source_type})`;
      select.appendChild(option);
    });

    this.enableUpdateSourceBtn();
  }

  async loadModels() {
    try {
      if (!this.projectId) {
        console.warn("No project ID available for loading models");
        return;
      }
      const response = await fetch(
        `${window.BACKEND_URL}/api/projects/${this.projectId}/models`,
      );
      if (!response.ok)
        throw new Error(`Failed to load models: ${response.status}`);

      this.models = await response.json();
      this.populateModelSelect();
    } catch (error) {
      console.error("Error loading models:", error);
      this.updateStatus(
        "modelStatus",
        `Error loading models: ${error.message}`,
        "error",
      );
    }
  }

  populateModelSelect() {
    const select = document.getElementById("modelSelect");
    select.innerHTML = '<option value="">Select a model...</option>';

    this.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.name;
      select.appendChild(option);
    });

    this.enableUpdateModelBtn();
  }

  async loadSinks() {
    try {
      const response = await fetch(`${window.BACKEND_URL}/api/sinks`);
      if (!response.ok)
        throw new Error(`Failed to load sinks: ${response.status}`);

      this.sinks = await response.json();
      this.populateSinkSelect();
    } catch (error) {
      console.error("Error loading sinks:", error);
      this.updateStatus(
        "sinkStatus",
        `Error loading sinks: ${error.message}`,
        "error",
      );
    }
  }

  populateSinkSelect() {
    const select = document.getElementById("sinkSelect");
    select.innerHTML = '<option value="">Select a sink...</option>';

    this.sinks.forEach((sink) => {
      const option = document.createElement("option");
      option.value = sink.id;
      option.textContent = `${sink.name} (${sink.sink_type})`;
      select.appendChild(option);
    });

    this.enableUpdateSinkBtn();
  }

  async loadPipeline() {
    try {
      console.log("Loading pipeline with projectId:", this.projectId);
      const url = `${window.BACKEND_URL}/api/projects/${this.projectId}/pipeline`;
      console.log("Fetching from URL:", url);

      const response = await fetch(url);
      if (!response.ok)
        throw new Error(`Failed to load pipeline: ${response.status}`);

      this.currentPipeline = await response.json();
      this.updatePipelineUI();
    } catch (error) {
      console.error("Error loading pipeline:", error);
      this.updateStatus(
        "pipelineStatus",
        `Error loading pipeline: ${error.message}`,
        "error",
      );
    }
  }

  updatePipelineUI() {
    if (this.currentPipeline) {
      // Update pipeline status
      const status = this.currentPipeline.status;
      const statusElement = document.getElementById("pipelineStatus");
      if (statusElement) {
        statusElement.textContent = `Status: ${status}`;
        statusElement.className =
          status === "RUNNING" ? "success-text" : "loading";
      }

      // Enable/disable buttons based on status
      const enableBtn = document.getElementById("enablePipelineBtn");
      const disableBtn = document.getElementById("disablePipelineBtn");
      if (enableBtn) enableBtn.disabled = status === "RUNNING";
      if (disableBtn) disableBtn.disabled = status !== "RUNNING";

      // Update source selection
      if (this.currentPipeline.source) {
        this.currentSource = this.currentPipeline.source;
        const sourceSelect = document.getElementById("sourceSelect");
        if (sourceSelect) sourceSelect.value = this.currentPipeline.source.id;
        this.updateStatus(
          "sourceStatus",
          `Source: ${this.currentPipeline.source.name}`,
          "success",
        );
      }

      // Update model selection
      if (this.currentPipeline.model) {
        this.currentModel = this.currentPipeline.model;
        const modelSelect = document.getElementById("modelSelect");
        if (modelSelect) modelSelect.value = this.currentPipeline.model.id;
        this.updateStatus(
          "modelStatus",
          `Model: ${this.currentPipeline.model.name}`,
          "success",
        );
      }

      // Update sink selection
      if (this.currentPipeline.sink) {
        this.currentSink = this.currentPipeline.sink;
        const sinkSelect = document.getElementById("sinkSelect");
        if (sinkSelect) sinkSelect.value = this.currentPipeline.sink.id;
        this.updateStatus(
          "sinkStatus",
          `Sink: ${this.currentPipeline.sink.name}`,
          "success",
        );
      }

      this.enableUpdateSourceBtn();
      this.enableUpdateModelBtn();
      this.enableUpdateSinkBtn();
    }
  }

  async enablePipeline() {
    try {
      const response = await fetch(
        `${window.BACKEND_URL}/api/projects/${this.projectId}/pipeline:enable`,
        {
          method: "POST",
        },
      );

      if (!response.ok)
        throw new Error(`Failed to enable pipeline: ${response.status}`);

      this.updateStatus("pipelineStatus", "Pipeline enabled", "success");
      const enableBtn = document.getElementById("enablePipelineBtn");
      const disableBtn = document.getElementById("disablePipelineBtn");
      if (enableBtn) enableBtn.disabled = true;
      if (disableBtn) disableBtn.disabled = false;

      // Reload pipeline to get updated status
      setTimeout(() => this.loadPipeline(), 1000);
    } catch (error) {
      console.error("Error enabling pipeline:", error);
      this.updateStatus("pipelineStatus", `Error: ${error.message}`, "error");
    }
  }

  async disablePipeline() {
    try {
      const response = await fetch(
        `${window.BACKEND_URL}/api/projects/${this.projectId}/pipeline:disable`,
        {
          method: "POST",
        },
      );

      if (!response.ok)
        throw new Error(`Failed to disable pipeline: ${response.status}`);

      this.updateStatus("pipelineStatus", "Pipeline disabled", "loading");
      const enableBtn = document.getElementById("enablePipelineBtn");
      const disableBtn = document.getElementById("disablePipelineBtn");
      if (enableBtn) enableBtn.disabled = false;
      if (disableBtn) disableBtn.disabled = true;

      // Reload pipeline to get updated status
      setTimeout(() => this.loadPipeline(), 1000);
    } catch (error) {
      console.error("Error disabling pipeline:", error);
      this.updateStatus("pipelineStatus", `Error: ${error.message}`, "error");
    }
  }

  async updateSource() {
    const selectedSourceId = document.getElementById("sourceSelect").value;

    if (!selectedSourceId) {
      this.updateStatus("sourceStatus", "Please select a source", "error");
      return;
    }

    this.updateStatus("sourceStatus", "Updating pipeline source...", "loading");

    try {
      // Update pipeline with selected source
      const pipelineResponse = await fetch(
        `${window.BACKEND_URL}/api/projects/${this.projectId}/pipeline`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source_id: selectedSourceId,
          }),
        },
      );

      if (!pipelineResponse.ok) {
        const errorData = await pipelineResponse.json();
        throw new Error(
          `Failed to update pipeline: ${errorData.detail || pipelineResponse.status}`,
        );
      }

      this.updateStatus(
        "sourceStatus",
        "Pipeline source updated successfully",
        "success",
      );

      // Reload pipeline
      setTimeout(() => this.loadPipeline(), 1000);
    } catch (error) {
      console.error("Error updating source:", error);
      this.updateStatus("sourceStatus", `Error: ${error.message}`, "error");
    }
  }

  async updateModel() {
    const modelId = document.getElementById("modelSelect").value;
    if (!modelId) return;

    this.updateStatus("modelStatus", "Updating pipeline model...", "loading");

    try {
      const response = await fetch(
        `${window.BACKEND_URL}/api/projects/${this.projectId}/pipeline`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: modelId }),
        },
      );

      if (!response.ok)
        throw new Error(`Failed to update model: ${response.status}`);

      this.updateStatus(
        "modelStatus",
        "Pipeline model updated successfully",
        "success",
      );

      // Reload pipeline
      setTimeout(() => this.loadPipeline(), 1000);
    } catch (error) {
      console.error("Error updating model:", error);
      this.updateStatus("modelStatus", `Error: ${error.message}`, "error");
    }
  }

  async updateSink() {
    const sinkId = document.getElementById("sinkSelect").value;
    if (!sinkId) return;

    this.updateStatus("sinkStatus", "Updating pipeline sink...", "loading");

    try {
      const response = await fetch(
        `${window.BACKEND_URL}/api/projects/${this.projectId}/pipeline`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sink_id: sinkId }),
        },
      );

      if (!response.ok)
        throw new Error(`Failed to update sink: ${response.status}`);

      this.updateStatus(
        "sinkStatus",
        "Pipeline sink updated successfully",
        "success",
      );

      // Reload pipeline
      setTimeout(() => this.loadPipeline(), 1000);
    } catch (error) {
      console.error("Error updating sink:", error);
      this.updateStatus("sinkStatus", `Error: ${error.message}`, "error");
    }
  }

  // Modal functions
  showAddSourceModal() {
    document.getElementById("addSourceModal").style.display = "block";
    this.resetModal();
  }

  hideAddSourceModal() {
    document.getElementById("addSourceModal").style.display = "none";
    this.modalStopCameraPreview();
  }

  resetModal() {
    document.getElementById("modalSourceName").value = "";
    document.getElementById("modalSourcePath").value = "";
    document.getElementById("modalCameraSelect").innerHTML =
      '<option value="">Click "Detect Cameras" first</option>';
    document.getElementById("modalCameraSelect").disabled = true;
    document.getElementById("modalSourceStatus").textContent =
      "Ready to create source";
    document.getElementById("modalSourceStatus").className = "loading";
    document.getElementById("createSourceBtn").disabled = true;
    this.toggleModalSourceType("webcam");
  }

  toggleModalSourceType(sourceType) {
    const cameraGroup = document.getElementById("modalCameraGroup");
    const fileGroup = document.getElementById("modalFileGroup");

    if (sourceType === "webcam") {
      cameraGroup.style.display = "block";
      fileGroup.style.display = "none";
    } else {
      cameraGroup.style.display = "none";
      fileGroup.style.display = "block";
    }

    this.enableModalCreateBtn();
  }

  async modalDetectCameras() {
    try {
      this.updateModalStatus("Detecting cameras...", "loading");
      await navigator.mediaDevices.getUserMedia({ video: true }); // Request permission
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput",
      );

      const cameraSelect = document.getElementById("modalCameraSelect");
      cameraSelect.innerHTML = '<option value="">Select a camera...</option>';

      videoDevices.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = index; // Use index for device_id
        option.textContent = device.label || `Camera ${index}`;
        cameraSelect.appendChild(option);
      });

      cameraSelect.disabled = false;
      this.updateModalStatus(
        `Found ${videoDevices.length} camera(s)`,
        "success",
      );
    } catch (error) {
      console.error("Error detecting cameras:", error);
      this.updateModalStatus(
        "Camera detection failed: " + error.message,
        "error",
      );
    }
  }

  async modalPreviewCamera(deviceIndex) {
    if (!deviceIndex) {
      this.modalStopCameraPreview();
      return;
    }

    try {
      this.modalStopCameraPreview(); // Stop any existing stream

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceIndex } },
      });

      const previewVideo = document.getElementById("modalPreviewVideo");
      previewVideo.srcObject = stream;
      previewVideo.play();

      const cameraPreview = document.getElementById("modalCameraPreview");
      cameraPreview.style.display = "block";

      this.updateModalStatus("Camera preview active", "success");
      this.enableModalCreateBtn();
    } catch (error) {
      console.error("Camera preview failed:", error);
      this.updateModalStatus(
        "Camera preview failed: " + error.message,
        "error",
      );
    }
  }

  modalStopCameraPreview() {
    const previewVideo = document.getElementById("modalPreviewVideo");
    if (previewVideo.srcObject) {
      previewVideo.srcObject.getTracks().forEach((track) => track.stop());
      previewVideo.srcObject = null;
    }

    const cameraPreview = document.getElementById("modalCameraPreview");
    cameraPreview.style.display = "none";
    this.updateModalStatus("Camera preview stopped", "loading");
  }

  enableModalCreateBtn() {
    const sourceType = document.getElementById("modalSourceType").value;
    const sourceName = document.getElementById("modalSourceName").value.trim();
    let canCreate = sourceName !== "";

    if (sourceType === "webcam") {
      const cameraSelect = document.getElementById("modalCameraSelect");
      canCreate = canCreate && cameraSelect.value !== "";
    } else {
      const sourcePath = document
        .getElementById("modalSourcePath")
        .value.trim();
      canCreate = canCreate && sourcePath !== "";
    }

    document.getElementById("createSourceBtn").disabled = !canCreate;
  }

  async createSource() {
    const sourceType = document.getElementById("modalSourceType").value;
    const sourceName = document.getElementById("modalSourceName").value.trim();

    if (!sourceName) {
      this.updateModalStatus("Please enter a source name", "error");
      return;
    }

    this.updateModalStatus("Creating source...", "loading");
    document.getElementById("createSourceBtn").disabled = true;

    try {
      let sourceData;

      if (sourceType === "webcam") {
        const deviceIndex = document.getElementById("modalCameraSelect").value;
        if (!deviceIndex) {
          this.updateModalStatus("Please select a camera", "error");
          return;
        }

        sourceData = {
          source_type: "webcam",
          name: sourceName,
          device_id: parseInt(deviceIndex),
        };
      } else {
        const sourcePath = document
          .getElementById("modalSourcePath")
          .value.trim();
        if (!sourcePath) {
          this.updateModalStatus("Please enter a file/folder path", "error");
          return;
        }

        sourceData = {
          source_type: sourceType,
          name: sourceName,
          [sourceType === "video_file" ? "video_path" : "folder_path"]:
            sourcePath,
        };
      }

      const response = await fetch(`${window.BACKEND_URL}/api/sources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sourceData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          `Failed to create source: ${errorData.detail || response.status}`,
        );
      }

      const newSource = await response.json();
      this.updateModalStatus(`Source created: ${newSource.name}`, "success");

      // Refresh sources list and close modal
      setTimeout(() => {
        this.loadSources();
        this.hideAddSourceModal();
      }, 1000);
    } catch (error) {
      console.error("Error creating source:", error);
      this.updateModalStatus(`Error: ${error.message}`, "error");
      document.getElementById("createSourceBtn").disabled = false;
    }
  }

  // Helper functions
  enableUpdateSourceBtn() {
    const sourceSelect = document.getElementById("sourceSelect");
    document.getElementById("updateSourceBtn").disabled = !sourceSelect.value;
  }

  enableUpdateModelBtn() {
    const modelSelect = document.getElementById("modelSelect");
    document.getElementById("updateModelBtn").disabled = !modelSelect.value;
  }

  enableUpdateSinkBtn() {
    const sinkSelect = document.getElementById("sinkSelect");
    document.getElementById("updateSinkBtn").disabled = !sinkSelect.value;
  }

  updateStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className =
      type === "success"
        ? "success-text"
        : type === "error"
          ? "error-text"
          : "loading";
  }

  updateModalStatus(message, type) {
    const element = document.getElementById("modalSourceStatus");
    element.textContent = message;
    element.className =
      type === "success"
        ? "success-text"
        : type === "error"
          ? "error-text"
          : "loading";
  }
}

// Initialize when page loads
document.addEventListener("DOMContentLoaded", () => {
  console.log("DOMContentLoaded event fired");
  console.log(
    "projectId element exists:",
    !!document.getElementById("projectId"),
  );
  window.pipelineController = new PipelineController();
});

// Also try immediate initialization as fallback
if (document.readyState === "loading") {
  console.log("DOM still loading, waiting for DOMContentLoaded");
} else {
  console.log("DOM already loaded, initializing immediately");
  console.log(
    "projectId element exists:",
    !!document.getElementById("projectId"),
  );
  if (!window.pipelineController) {
    window.pipelineController = new PipelineController();
  }
}
