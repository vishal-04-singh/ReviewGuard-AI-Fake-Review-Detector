// ============================================================
// ReviewGuard – Content Script
// Runs on e-commerce pages. Detects, extracts, analyses reviews
// and injects visual badges directly into the DOM.
// ============================================================

(function () {
  "use strict";

  // ---- Site-specific CSS selectors ----
  const SITE_SELECTORS = {
    "amazon.in":  { container: "[data-hook='review']",    text: "[data-hook='review-body'] span", rating: "[data-hook='review-star-rating'] span", author: ".a-profile-name" },
    "amazon.com": { container: "[data-hook='review']",    text: "[data-hook='review-body'] span", rating: "[data-hook='review-star-rating'] span", author: ".a-profile-name" },
    "flipkart.com": { container: "._27M-vq",              text: "._6K-7Co",                       rating: "._3LWZlK",                              author: "._2V5EHH ._2NsDsF" },
    "meesho.com":  { container: ".ReviewCard__Container", text: ".ReviewCard__Text",              rating: ".Rating__Container",                     author: ".ReviewCard__Name" },
    "myntra.com":  { container: ".user-review-main",      text: ".user-review-reviewTextWrapper", rating: ".user-review-ratingBadge",               author: ".user-review-left-name" },
    "snapdeal.com":{ container: ".reviewbox",             text: ".reviewDescription",             rating: ".filled-stars",                          author: ".reviewer-name" }
  };

  const BACKEND_URL = "http://localhost:5000/classify";
  let pageResults  = { total: 0, suspicious: 0, genuine: 0, pending: 0 };
  let analysisData = [];

  // ---- Detect which site we're on ----
  function getSiteSelectors() {
    const host = window.location.hostname;
    for (const domain in SITE_SELECTORS) {
      if (host.includes(domain)) return { selectors: SITE_SELECTORS[domain], domain };
    }
    return null;
  }

  // ---- Extract review text from a container element ----
  function extractReviewData(container, selectors) {
    const textEl   = container.querySelector(selectors.text);
    const ratingEl = container.querySelector(selectors.rating);
    const authorEl = container.querySelector(selectors.author);

    const text   = textEl   ? textEl.innerText.trim()   : "";
    const rating = ratingEl ? parseFloat(ratingEl.innerText) || 0 : 0;
    const author = authorEl ? authorEl.innerText.trim() : "Anonymous";

    return { text, rating, author };
  }

  // ---- Inline heuristic classifier (offline fallback) ----
  function heuristicClassify(reviewData) {
    const { text, rating } = reviewData;
    let suspicionScore = 0;
    const flags = [];

    const words = text.split(/\s+/).filter(Boolean);

    // Too short
    if (words.length < 8) { suspicionScore += 30; flags.push("Very short review"); }
    else if (words.length < 15) { suspicionScore += 10; flags.push("Short review"); }

    // Excessive capitalisation
    const capsRatio = (text.match(/[A-Z]/g) || []).length / Math.max(text.length, 1);
    if (capsRatio > 0.3) { suspicionScore += 20; flags.push("Excessive caps"); }

    // Superlative / spammy words
    const spammyPhrases = [
      "best ever", "greatest ever", "perfect product", "100% recommend",
      "must buy", "amazing quality", "super quality", "5 star", "highly recommend",
      "life changing", "absolutely love", "best purchase", "best product ever"
    ];
    const lower = text.toLowerCase();
    spammyPhrases.forEach(p => {
      if (lower.includes(p)) { suspicionScore += 15; flags.push(`Spammy phrase: "${p}"`); }
    });

    // Repeated words
    const wordFreq = {};
    words.forEach(w => { const wl = w.toLowerCase().replace(/[^a-z]/g, ""); wordFreq[wl] = (wordFreq[wl] || 0) + 1; });
    const maxFreq = Math.max(...Object.values(wordFreq));
    if (maxFreq > 4 && words.length < 40) { suspicionScore += 15; flags.push("Repeated words"); }

    // Generic product-spec dump (numbers + units without opinion)
    const specPattern = /\d+\s*(gb|mb|mp|hz|mah|inch|cm|mm|kg|watt|w)\b/gi;
    const specMatches = (text.match(specPattern) || []).length;
    if (specMatches > 3 && words.length < 30) { suspicionScore += 20; flags.push("Spec dump without opinion"); }

    // Extremely high rating with no cons
    const conWords = ["but", "however", "although", "except", "downside", "issue", "problem", "con", "negative", "bad", "disappoint", "unfortunately"];
    const hasCon = conWords.some(w => lower.includes(w));
    if (rating >= 5 && !hasCon && words.length < 30) { suspicionScore += 15; flags.push("5-star, no cons mentioned"); }

    // No personal experience
    const personalWords = ["i", "my", "me", "we", "our", "myself", "bought", "ordered", "received", "used", "using"];
    const hasPersonal = personalWords.some(w => lower.split(/\s+/).includes(w));
    if (!hasPersonal) { suspicionScore += 10; flags.push("No personal experience words"); }

    // Sentiment vs rating mismatch (very negative text + high stars)
    const negWords = ["terrible", "awful", "horrible", "worst", "hate", "useless", "broke", "broken", "waste", "scam", "fake"];
    const hasNeg = negWords.some(w => lower.includes(w));
    if (hasNeg && rating >= 4) { suspicionScore += 25; flags.push("Negative text but high rating"); }

    const confidence = Math.min(suspicionScore, 100);
    const isSuspicious = confidence >= 35;

    return {
      label: isSuspicious ? "Suspicious" : "Genuine",
      confidence: isSuspicious ? confidence : 100 - confidence,
      flags,
      method: "heuristic"
    };
  }

  // ---- Try backend, fall back to heuristic ----
  async function classifyReview(reviewData) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);
      const resp = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: reviewData.text, rating: reviewData.rating }),
        signal: controller.signal
      });
      clearTimeout(timeout);
      if (resp.ok) {
        const data = await resp.json();
        return { ...data, method: "ml" };
      }
    } catch (_) { /* fall through */ }
    return heuristicClassify(reviewData);
  }

  // ---- Inject badge into review card ----
  function injectBadge(container, result) {
    // Remove any existing badge
    const old = container.querySelector(".rg-badge");
    if (old) old.remove();

    const badge = document.createElement("div");
    badge.className = "rg-badge";
    badge.setAttribute("data-label", result.label);

    const isSuspicious = result.label === "Suspicious";
    badge.style.cssText = `
      display:inline-flex; align-items:center; gap:6px;
      padding:4px 10px; border-radius:20px; font-size:12px;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      font-weight:600; letter-spacing:0.3px; margin-top:8px;
      background:${isSuspicious ? "#ff4d4d22" : "#00e67622"};
      border:1px solid ${isSuspicious ? "#ff4d4d" : "#00e676"};
      color:${isSuspicious ? "#ff4d4d" : "#00e676"};
      cursor:pointer; position:relative; z-index:9999;
    `;

    const icon = isSuspicious ? "⚠️" : "✅";
    const conf = Math.round(result.confidence);
    badge.innerHTML = `
      <span>${icon}</span>
      <span>${result.label}</span>
      <span style="opacity:0.7;font-weight:400">${conf}% confidence</span>
      ${result.method === "ml" ? '<span style="opacity:0.5;font-size:10px">ML</span>' : ''}
    `;

    // Tooltip with flags
    if (result.flags && result.flags.length > 0) {
      const tip = document.createElement("div");
      tip.style.cssText = `
        display:none; position:absolute; bottom:calc(100% + 6px); left:0;
        background:#1a1a1a; border:1px solid #333; border-radius:8px;
        padding:8px 12px; font-size:11px; color:#ccc; min-width:200px;
        max-width:280px; z-index:99999; line-height:1.6; font-weight:400;
        box-shadow:0 4px 20px rgba(0,0,0,0.5);
      `;
      tip.innerHTML = "<strong style='color:#fff'>Flags detected:</strong><br>" +
        result.flags.map(f => `• ${f}`).join("<br>");
      badge.appendChild(tip);

      badge.addEventListener("mouseenter", () => { tip.style.display = "block"; });
      badge.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    }

    // Insert badge after the review text or at end of container
    const textEl = container.querySelector("[data-hook='review-body'] span, ._6K-7Co, .ReviewCard__Text, .user-review-reviewTextWrapper, .reviewDescription");
    if (textEl) {
      textEl.parentNode.insertAdjacentElement("afterend", badge);
    } else {
      container.appendChild(badge);
    }

    // Highlight container if suspicious
    if (isSuspicious) {
      container.style.cssText += ";border-left:3px solid #ff4d4d !important; padding-left:8px !important;";
    }
  }

  // ---- Main analysis function ----
  async function analyseReviews() {
    const site = getSiteSelectors();
    if (!site) return;

    const containers = document.querySelectorAll(site.selectors.container);
    if (!containers.length) return;

    pageResults = { total: containers.length, suspicious: 0, genuine: 0, pending: containers.length };
    analysisData = [];

    // Notify popup of start
    chrome.runtime.sendMessage({ type: "ANALYSIS_STARTED", total: containers.length });

    for (let i = 0; i < containers.length; i++) {
      const container = containers[i];
      const reviewData = extractReviewData(container, site.selectors);

      if (!reviewData.text || reviewData.text.length < 3) {
        pageResults.pending--;
        continue;
      }

      const result = await classifyReview(reviewData);
      result.text   = reviewData.text.substring(0, 120) + (reviewData.text.length > 120 ? "…" : "");
      result.author = reviewData.author;
      result.rating = reviewData.rating;

      injectBadge(container, result);
      analysisData.push(result);

      if (result.label === "Suspicious") pageResults.suspicious++;
      else pageResults.genuine++;
      pageResults.pending--;

      // Send live update
      chrome.runtime.sendMessage({ type: "REVIEW_CLASSIFIED", result, stats: { ...pageResults }, index: i });
    }

    chrome.runtime.sendMessage({ type: "ANALYSIS_COMPLETE", stats: { ...pageResults }, reviews: analysisData });
    chrome.storage.local.set({ lastStats: pageResults, lastReviews: analysisData, lastUrl: window.location.href });
  }

  // ---- Listen for messages from popup / background ----
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type === "GET_STATUS") {
      sendResponse({ stats: pageResults, reviews: analysisData, url: window.location.href });
    }
    if (msg.type === "REANALYSE") {
      analyseReviews();
      sendResponse({ ok: true });
    }
  });

  // ---- Observe DOM changes (infinite scroll / lazy loading) ----
  let debounceTimer = null;
  const observer = new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const site = getSiteSelectors();
      if (!site) return;
      const unbadged = [...document.querySelectorAll(site.selectors.container)]
        .filter(c => !c.querySelector(".rg-badge"));
      if (unbadged.length > 0) analyseReviews();
    }, 1500);
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // Kick off initial analysis after page settles
  setTimeout(analyseReviews, 2000);
})();
