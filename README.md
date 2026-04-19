# 🛡️ ReviewGuard – AI Fake Review Detector

**ReviewGuard** is a Chrome browser extension that automatically detects fake and suspicious product reviews on e-commerce platforms using NLP + Machine Learning.

## Project by
- Vishal Singh (590028030)
- Vivek Rawat (590027033)
- Ankit Kumar Singh (590027049)

---

## Supported Sites
| Platform | Status |
|---|---|
| Amazon.in / Amazon.com | ✅ Full support |
| Flipkart | ✅ Full support |
| Meesho | ✅ Full support |
| Myntra | ✅ Full support |
| Snapdeal | ✅ Full support |

---

## Architecture Overview

```
User Browser
    │
    ▼
Browser Extension (content.js)
    │  extracts review DOM
    ▼
Background Script (background.js)
    │  badge counter
    ▼
Popup UI (popup.html + popup.js)
    │
    ▼ HTTP POST
Flask API (app.py) ──► NLP preprocessing
    │                ──► TF-IDF vectorization
    │                ──► LinearSVC classification
    │                ──► Confidence + flag generation
    ▼
Classification Result → Injected badge on page
```

---

## Quick Start

### 1. Install the Extension

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer Mode** (top right)
3. Click **Load Unpacked**
4. Select the `extension/` folder

### 2. Set Up the ML Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Train the ML model using the fake reviews dataset
python model_trainer.py

# Start the API server
python app.py
```

The server runs on **http://localhost:5000**

**Training Data:** The model is trained on `fake reviews dataset.csv` (~40K labeled reviews from Amazon)

### 3. Use ReviewGuard

1. Visit any supported e-commerce product page
2. The extension auto-scans reviews within 2 seconds
3. Click the extension icon to see the summary popup
4. Each review gets a coloured badge: **✅ Genuine** or **⚠️ Suspicious**
5. Hover over a badge to see why it was flagged

---

## ML Pipeline

| Component | Detail |
|---|---|
| Training Data | 40,427 labeled reviews (fake reviews dataset.csv) |
| Feature Extraction | TF-IDF (unigrams, bigrams, trigrams) |
| Max Features | 8,000 vocabulary tokens |
| Algorithm | LinearSVC (Calibrated for probabilities) |
| Test Accuracy | ~94.8% |
| Problem Type | Binary Classification (0=Genuine, 1=Suspicious) |
| Offline Fallback | Heuristic rule engine (no server required) |

### Heuristic Flags Detected
- Very short review (< 8 words)
- Excessive capitalisation
- Spammy/superlative phrases ("best ever", "must buy")
- Repeated words
- No personal pronouns or experience words
- High rating with no cons mentioned
- Negative language with high star rating
- Spec dump without personal opinion

---

## API Endpoints

### `POST /classify`
Classify a single review.

**Request:**
```json
{
  "text": "Best product ever! Amazing quality!! Must buy!!",
  "rating": 5
}
```

**Response:**
```json
{
  "label": "Suspicious",
  "confidence": 94.3,
  "flags": ["Spammy phrase: \"best product ever\"", "No personal experience words"],
  "word_count": 9,
  "method": "ml"
}
```

### `POST /classify/bulk`
Classify up to 100 reviews at once.

**Request:**
```json
{
  "reviews": [
    { "text": "...", "rating": 5 },
    { "text": "...", "rating": 3 }
  ]
}
```

### `GET /health`
Returns `{ "status": "ok", "model_loaded": true }`

### `GET /model/info`
Returns model accuracy and metadata.

---

## Database Design (from system design doc)

| Table | Fields |
|---|---|
| Users | user_id (PK), name, email |
| Reviews | review_id (PK), product_id (FK), review_text, timestamp |
| Classification | review_id (FK), classification, confidence_score |
| Model Metadata | model_version, accuracy, training_date |

---

## File Structure

```
reviewguard/
├── extension/
│   ├── manifest.json       ← Chrome Extension config
│   ├── content.js          ← DOM extraction + badge injection
│   ├── background.js       ← Service worker + badge counter
│   ├── popup.html          ← Extension popup UI
│   ├── popup.js            ← Popup logic + live updates
│   ├── settings.html       ← Settings page
│   └── icons/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
└── backend/
    ├── app.py              ← Flask REST API
    ├── model_trainer.py    ← ML training pipeline
    ├── model.pkl           ← Trained model (generated)
    ├── model_meta.json     ← Model metadata (generated)
    ├── fake reviews dataset.csv  ← Training data (~40K reviews)
    └── requirements.txt
```

---

## Performance Goals
- Response time < 2 seconds per review
- Handles thousands of concurrent reviews
- Offline heuristic mode when backend unavailable
- Real-time DOM observation for infinite-scroll pages
