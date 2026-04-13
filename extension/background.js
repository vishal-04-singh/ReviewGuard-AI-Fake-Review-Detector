// ============================================================
// ReviewGuard – Background Service Worker
// Manages extension state and badge counter
// ============================================================

chrome.runtime.onInstalled.addListener(() => {
  console.log("ReviewGuard installed.");
  chrome.storage.local.set({ backendUrl: "http://localhost:5000", enabled: true });
});

// Update badge count on toolbar icon
chrome.runtime.onMessage.addListener((msg, sender) => {
  const tabId = sender.tab?.id;
  if (!tabId) return;

  if (msg.type === "ANALYSIS_STARTED") {
    chrome.action.setBadgeText({ text: "…", tabId });
    chrome.action.setBadgeBackgroundColor({ color: "#888888", tabId });
  }

  if (msg.type === "REVIEW_CLASSIFIED" || msg.type === "ANALYSIS_COMPLETE") {
    const { suspicious, total } = msg.stats;
    if (suspicious > 0) {
      chrome.action.setBadgeText({ text: String(suspicious), tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#ff4d4d", tabId });
    } else if (msg.type === "ANALYSIS_COMPLETE") {
      chrome.action.setBadgeText({ text: "✓", tabId });
      chrome.action.setBadgeBackgroundColor({ color: "#00c853", tabId });
    }
  }
});
