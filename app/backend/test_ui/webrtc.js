import { BACKEND_URL } from "./api.js";

const videoEl = document.getElementById("remoteVideo");
const connectBtn = document.getElementById("connectBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const statsEl = document.getElementById("statsOverlay");
const videoMain = document.getElementById("videoMain");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const autoToggle = document.getElementById("autoConnectToggle");

let pc = null;
let isConnected = false;
let isManualDisconnect = false;
let statsTimer = null;
let reconnectTimer = null;
let lastStats = null;
let isConnecting = false; // retained for button state only

let webrtcId = localStorage.getItem("anomalib_test_ui_webrtc_id");
if (!webrtcId) {
  webrtcId = Math.random().toString(36).slice(2);
  localStorage.setItem("anomalib_test_ui_webrtc_id", webrtcId);
}

function toast(msg, type = "info") {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.className = `toast show toast-${type}`;
  window.clearTimeout(el._t);
  el._t = window.setTimeout(() => {
    el.textContent = "";
    el.className = "toast hidden";
  }, 2500);
}

function setButtons() {
  connectBtn.disabled = isConnected || isConnecting;
  disconnectBtn.disabled = !isConnected;
}

function clearTimers() {
  if (statsTimer) {
    clearInterval(statsTimer);
    statsTimer = null;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function stopStream() {
  clearTimers();
  if (pc) {
    try {
      pc.close();
    } catch {}
  }
  pc = null;
  isConnected = false;
  isConnecting = false;
  if (videoEl) videoEl.srcObject = null;
  setButtons();
}

async function startStatsLoop() {
  if (!pc || !pc.getStats) return;
  lastStats = null;
  statsTimer = setInterval(async () => {
    try {
      const report = await pc.getStats(null);
      let inbound;
      report.forEach((r) => {
        if (
          (r.type === "inbound-rtp" || r.type === "inbound-rtp-media") &&
          r.kind === "video"
        )
          inbound = r;
      });
      if (!inbound) return;
      const now = performance.now();
      if (lastStats) {
        const dt = (now - lastStats.t) / 1000;
        const dBytes = (inbound.bytesReceived || 0) - (lastStats.bytes || 0);
        const dFrames = (inbound.framesDecoded || 0) - (lastStats.frames || 0);
        const kbps = Math.max(0, Math.round((dBytes * 8) / dt / 1000));
        const fps = Math.max(0, Math.round(dFrames / dt));
        if (statsEl) statsEl.textContent = `${kbps} kbps · ${fps} fps`;
      }
      lastStats = {
        t: now,
        bytes: inbound.bytesReceived || 0,
        frames: inbound.framesDecoded || 0,
      };
    } catch {
      /* ignore */
    }
  }, 1000);
}

function scheduleReconnect() {
  if (isManualDisconnect) return;
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, 2000);
}

async function connect() {
  isManualDisconnect = false;
  stopStream();
  setButtons();
  toast("Connecting to WebRTC...", "info");
  try {
    isConnecting = true;
    setButtons();
    pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });
    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (event) => {
      if (event.track.kind === "video") {
        videoEl.srcObject = event.streams[0];
        // clear connecting state once we start receiving video
        const hide = () => {
          isConnecting = false;
          setButtons();
        };
        hide();
        videoEl.addEventListener("loadeddata", hide, { once: true });
        videoEl.addEventListener("playing", hide, { once: true });
      }
    };

    pc.onconnectionstatechange = () => {
      const s = pc.connectionState;
      if (s === "connected") {
        isConnected = true;
        isConnecting = false;
        setButtons();
        startStatsLoop();
        toast("Stream connected", "success");
      } else if (s === "failed" || s === "disconnected") {
        isConnected = false;
        isConnecting = false;
        setButtons();
        toast("Stream disconnected", "error");
        clearTimers();
        scheduleReconnect();
      }
    };

    const offer = await pc.createOffer({ offerToReceiveVideo: true });
    await pc.setLocalDescription(offer);

    // wait for ICE gathering complete to include candidates in SDP
    await new Promise((resolve) => {
      if (pc.iceGatheringState === "complete") {
        resolve();
      } else {
        const checkState = () => {
          if (pc.iceGatheringState === "complete") {
            pc.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }
        };
        pc.addEventListener("icegatheringstatechange", checkState);
        // safety timeout
        setTimeout(() => {
          pc.removeEventListener("icegatheringstatechange", checkState);
          resolve();
        }, 1500);
      }
    });

    const res = await fetch(`${BACKEND_URL}/api/webrtc/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        webrtc_id: webrtcId,
        sdp: offer.sdp,
        type: offer.type,
      }),
    });
    if (!res.ok) throw new Error(`Offer failed: ${res.status}`);
    const answer = await res.json();
    const desc =
      typeof answer?.sdp === "string" && answer?.type
        ? answer
        : { sdp: answer, type: "answer" };
    await pc.setRemoteDescription(new RTCSessionDescription(desc));
  } catch (e) {
    console.error("WebRTC error:", e);
    toast(`WebRTC error: ${e.message}`, "error");
    isConnected = false;
    isConnecting = false;
    setButtons();
    scheduleReconnect();
  }
}

function disconnect() {
  isManualDisconnect = true;
  stopStream();
  toast("Disconnected", "info");
}

connectBtn.addEventListener("click", connect);
disconnectBtn.addEventListener("click", disconnect);

// Fullscreen helpers
function updateFullscreenButton() {
  if (!fullscreenBtn) return;
  fullscreenBtn.textContent = document.fullscreenElement
    ? "Exit Fullscreen"
    : "Fullscreen";
}
async function requestAnyFullscreen() {
  try {
    if (videoMain && videoMain.requestFullscreen)
      return await videoMain.requestFullscreen();
  } catch {}
  try {
    // Fallback to video element
    if (videoEl && videoEl.requestFullscreen)
      return await videoEl.requestFullscreen();
  } catch {}
  try {
    // Safari iOS presentation mode
    if (
      videoEl &&
      videoEl.webkitSupportsPresentationMode &&
      typeof videoEl.webkitSetPresentationMode === "function"
    ) {
      videoEl.webkitSetPresentationMode("fullscreen");
      return;
    }
    if (videoEl && videoEl.webkitEnterFullScreen) {
      videoEl.webkitEnterFullScreen();
      return;
    }
  } catch {}
  throw new Error("Fullscreen not supported");
}
if (fullscreenBtn) {
  fullscreenBtn.addEventListener("click", async () => {
    try {
      if (!document.fullscreenElement) {
        await requestAnyFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (e) {
      toast(`Fullscreen error: ${e.message}`, "error");
    } finally {
      updateFullscreenButton();
    }
  });
  document.addEventListener("fullscreenchange", updateFullscreenButton);
  updateFullscreenButton();
}

// Auto-connect preference
const savedAuto = localStorage.getItem("anomalib_auto_connect") === "1";
if (autoToggle) {
  autoToggle.checked = savedAuto;
  autoToggle.addEventListener("change", () => {
    localStorage.setItem(
      "anomalib_auto_connect",
      autoToggle.checked ? "1" : "0",
    );
    if (autoToggle.checked && !isConnected && !isConnecting) connect();
  });
  if (savedAuto) {
    setTimeout(() => {
      if (!isConnected && !isConnecting) connect();
    }, 200);
  }
}

// Expose controls for other modules (tabs) to manage WebRTC
window.webrtcConnect = connect;
window.webrtcDisconnect = disconnect;
