// ============================================================
// ReviewGuard – Popup Script
// ============================================================

(function () {
  "use strict";

  // DOM refs
  const statTotal      = document.getElementById("statTotal");
  const statSuspicious = document.getElementById("statSuspicious");
  const statGenuine    = document.getElementById("statGenuine");
  const progressFill   = document.getElementById("progressFill");
  const progressLabel  = document.getElementById("progressLabel");
  const progressPct    = document.getElementById("progressPct");
  const reviewsList    = document.getElementById("reviewsList");
  const alertBanner    = document.getElementById("alertBanner");
  const statusDot      = document.getElementById("statusDot");
  const methodTag      = document.getElementById("methodTag");
  const listHdr        = document.getElementById("listHdr");
  const btnRescan      = document.getElementById("btnRescan");
  const btnSettings    = document.getElementById("btnSettings");

  let allReviews = [];
  let currentStats = {};

  // ---- Helper: update stats ----
  function updateStats(stats) {
    currentStats = stats;
    statTotal.textContent      = stats.total      || 0;
    statSuspicious.textContent = stats.suspicious || 0;
    statGenuine.textContent    = stats.genuine    || 0;

    const done = (stats.total || 0) - (stats.pending || 0);
    const pct  = stats.total ? Math.round((done / stats.total) * 100) : 0;
    progressFill.style.width = pct + "%";
    progressPct.textContent  = pct + "%";

    if (stats.pending > 0) {
      progressLabel.innerHTML = `<span class="spinning">⚙</span> Analysing… (${done}/${stats.total})`;
    } else if (stats.total > 0) {
      progressLabel.textContent = `Analysis complete – ${stats.total} reviews scanned`;
      statusDot.classList.remove("inactive");
    }
  }

  // ---- Helper: render review list ----
  function renderReviews(reviews) {
    if (!reviews || reviews.length === 0) return;
    allReviews = reviews;
    listHdr.style.display = "block";

    // Sort: suspicious first
    const sorted = [...reviews].sort((a, b) => {
      if (a.label === "Suspicious" && b.label !== "Suspicious") return -1;
      if (b.label === "Suspicious" && a.label !== "Suspicious") return 1;
      return (b.confidence || 0) - (a.confidence || 0);
    });

    reviewsList.innerHTML = "";
    sorted.forEach(r => {
      const isSusp = r.label === "Suspicious";
      const el = document.createElement("div");
      el.className = `review-item ${isSusp ? "suspicious" : "genuine"}`;

      const flagsHtml = (r.flags && r.flags.length)
        ? `<div class="review-flags">${r.flags.slice(0, 3).map(f => `<span class="flag-chip">${f}</span>`).join("")}</div>`
        : "";

      const conf = Math.round(r.confidence || 0);

      el.innerHTML = `
        <div class="review-top">
          <span class="review-author">${escHtml(r.author || "Anonymous")}</span>
          <span class="review-badge ${isSusp ? "suspicious" : "genuine"}">${r.label}</span>
        </div>
        <div class="review-text">${escHtml(r.text || "")}</div>
        ${flagsHtml}
        <div class="confidence-bar-wrap">
          <div class="confidence-bar ${isSusp ? "suspicious" : "genuine"}" style="width:${conf}%"></div>
        </div>
      `;
      reviewsList.appendChild(el);
    });

    // Set method tag
    const usedML = reviews.some(r => r.method === "ml");
    methodTag.style.display = "inline-flex";
    if (usedML) {
      methodTag.className = "method-tag ml";
      methodTag.textContent = "🤖 ML Model active";
    } else {
      methodTag.className = "method-tag heur";
      methodTag.textContent = "⚡ Heuristic mode (start backend for ML)";
    }

    // Alert banner
    const suspicious = reviews.filter(r => r.label === "Suspicious").length;
    if (suspicious > 0) {
      showAlert(`⚠️ ${suspicious} suspicious review${suspicious > 1 ? "s" : ""} found on this page!`, "warning");
    } else if (reviews.length > 0) {
      showAlert("✅ All reviews appear genuine.", "success");
    }
  }

  function escHtml(str) {
    return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }

  function showAlert(msg, type) {
    alertBanner.className = `alert-banner show ${type}`;
    alertBanner.innerHTML = msg;
  }

  // ---- Load from storage (persisted results) ----
  function loadFromStorage() {
    chrome.storage.local.get(["lastStats", "lastReviews", "lastUrl"], (data) => {
      if (data.lastStats) updateStats(data.lastStats);
      if (data.lastReviews && data.lastReviews.length) renderReviews(data.lastReviews);
    });
  }

  // ---- Query active tab content script ----
  function queryActiveTab() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]) return;
      const tab = tabs[0];

      const supportedHosts = ["amazon.in","amazon.com","flipkart.com","meesho.com","myntra.com","snapdeal.com"];
      const isSupported = supportedHosts.some(h => tab.url && tab.url.includes(h));

      if (!isSupported) {
        showAlert("ℹ️ Navigate to a supported e-commerce site to use ReviewGuard.", "info");
        return;
      }

      chrome.tabs.sendMessage(tab.id, { type: "GET_STATUS" }, (resp) => {
        if (chrome.runtime.lastError || !resp) {
          loadFromStorage();
          return;
        }
        if (resp.stats) updateStats(resp.stats);
        if (resp.reviews) renderReviews(resp.reviews);
      });
    });
  }

  // ---- Listen for live updates ----
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "ANALYSIS_STARTED") {
      reviewsList.innerHTML = '<div class="empty-state"><div class="emoji spinning">⚙️</div><p>Analysing reviews…</p></div>';
      listHdr.style.display = "none";
      alertBanner.className = "alert-banner";
    }
    if (msg.type === "REVIEW_CLASSIFIED") {
      updateStats(msg.stats);
    }
    if (msg.type === "ANALYSIS_COMPLETE") {
      updateStats(msg.stats);
      renderReviews(msg.reviews);
    }
  });

  // ---- Buttons ----
  btnRescan.addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "REANALYSE" });
        reviewsList.innerHTML = '<div class="empty-state"><div class="emoji spinning">⚙️</div><p>Re-scanning…</p></div>';
        listHdr.style.display = "none";
        alertBanner.className = "alert-banner";
      }
    });
  });

  btnSettings.addEventListener("click", () => {
    chrome.tabs.create({ url: "settings.html" });
  });

  // ---- Init ----
  queryActiveTab();
})();
