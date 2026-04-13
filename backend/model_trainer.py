"""
ReviewGuard – Model Trainer
Trains a TF-IDF + Classifier pipeline on labeled review data.
Run this once before starting the API server: python model_trainer.py
"""

import pickle, os, json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import re

# ----------------------------------------------------------------
# Training data  (0 = Genuine, 1 = Suspicious/Fake)
# ----------------------------------------------------------------
TRAINING_DATA = [
    # ---- GENUINE (0) ----
    ("I bought this phone last month and have been using it daily. The battery lasts a full day with moderate usage, camera takes decent pictures in daylight but struggles in low light. Build quality feels solid. The only downside is the heating when playing games.", 0),
    ("This laptop exceeded my expectations. The display is crisp and the keyboard is comfortable for long typing sessions. Boot time is fast and it handles multitasking well. However, the fan gets quite loud under heavy load. Overall great value for the price.", 0),
    ("Used this for three months now. The sound quality is excellent for the price range. Bass is punchy and highs are clear. Comfort is good for up to 2 hours but starts to feel tight after that. The cable is a bit short. Would recommend for casual listening.", 0),
    ("Received the product in good condition. Instructions were clear and setup took about 20 minutes. Works as described. Had a minor issue with the app connectivity but customer support was helpful. Using it daily now with no further issues.", 0),
    ("Average product honestly. Does the job but nothing special. The material feels a bit cheap compared to the price, and one of the stitches came undone after two weeks of light use. Works fine for basic needs but I expected better quality.", 0),
    ("Been using this for 6 months. The performance is consistent and hasn't slowed down. Battery still holds a good charge. Screen is vibrant. Minor annoyance: the charger gets warm. But overall I'm happy with my purchase and would buy again.", 0),
    ("Not exactly what I expected. The color looks different from the photos – more of an off-white than pure white. Functionality is fine though. Size is accurate to the description. I'd rate it 3/5 for meeting basic needs but missing the expected premium feel.", 0),
    ("Great budget option. I was skeptical at first but after two weeks of daily use I'm impressed. The build is sturdy, performance is smooth for everyday tasks. The camera is average but acceptable for the price. Not perfect but excellent value.", 0),
    ("The delivery was quick and packaging was secure. The item itself is exactly as described. I've been using it for a week and it works flawlessly. The design is sleek. My only concern is long-term durability – time will tell. So far so good.", 0),
    ("Mixed feelings about this. Good build quality and nice design. Performance is solid for regular use. But the battery life is disappointing – barely lasts half a day with normal use. For the price I expected better. Still usable but not great.", 0),
    ("This is my second one – bought a replacement after the first lasted 2 years. Same great quality. The hinge mechanism is smooth, storage is adequate, and the display is easy on the eyes. Slightly pricier than before but still worth it.", 0),
    ("Ordered this as a gift and the recipient loved it. The packaging was premium and the product looked exactly like the listing. Works perfectly. Minor gripe: one of the buttons is a bit stiff initially but loosens up with use.", 0),
    ("Genuinely impressed. I was hesitant because of some negative reviews but my experience has been positive. Setup was straightforward, performance is great. The only issue is occasional lag when switching between apps, which is minor. Recommended.", 0),
    ("Decent product for the price. Not the best I've used but definitely not the worst. Does what it claims. Customer service was responsive when I had a question. Build could be more robust. I'd give it 3.5 stars if possible.", 0),
    ("Love the design but functionality is hit or miss. Works great 80% of the time, but occasionally acts glitchy. Had to restart it twice in the first week. Response from support was slow. I'll update review if things improve or worsen.", 0),

    # ---- SUSPICIOUS / FAKE (1) ----
    ("BEST PRODUCT EVER!!! Amazing quality!! Must buy!! 100% recommend to everyone!! You will love it so much!! Best purchase of my life!! 5 stars always!!", 1),
    ("Good product good quality good price good delivery good packaging. Very good. Buy now.", 1),
    ("Best product. Amazing. Superb. Excellent. Outstanding. Perfect. Brilliant. Wonderful. Buy this.", 1),
    ("Product is very good. I like it. Quality is good. Fast delivery. Good packing. Nice product. Happy with purchase.", 1),
    ("Nice product very nice quality nice delivery nice packing very nice product very nice.", 1),
    ("WOW!!! Just received! AMAZING product! SUPER FAST delivery! BEST SELLER! BUY NOW! 10/10! HIGHLY RECOMMEND!", 1),
    ("This is the greatest phone ever. 6GB RAM 128GB storage 48MP camera 5000mAh battery best phone. Must buy now.", 1),
    ("Excellent product. Very good quality. Fast shipping. 5 star. Recommend. Good price. Will buy again. Thank you.", 1),
    ("PERFECT PRODUCT. BEST QUALITY. FAST DELIVERY. NICE PACKING. 100% ORIGINAL. MUST BUY. BEST SELLER. 5 STAR.", 1),
    ("Very good product I love it so much best product ever amazing quality best price best delivery best packing best best best.", 1),
    ("Product received. Good. Will recommend to all. Buy now. 5 star rating. Very nice quality packaging.", 1),
    ("Super product. Ultra quality. Mega fast delivery. 5 star product. Buy immediately. Best in market. No issues.", 1),
    ("I love this product it is so amazing and wonderful and perfect and I recommend it to everyone who wants the best product.", 1),
    ("Best headphones ever!!! 40mm drivers 30hr battery bluetooth 5.0 noise cancellation. Amazing sound. Buy now!!", 1),
    ("Wow what a product! Speechless! Nothing to complain about! Pure perfection! Zero flaws! 5 stars forever!", 1),
    ("Good product fast delivery good quality satisfied customer recommend buying must buy five star rating thank you seller.", 1),
    ("Superb!! Excellent!! Wonderful!! Brilliant!! Magnificent!! Outstanding!! Buy immediately!! Best product ever!!", 1),
    ("Nice product. Good quality. Fast shipping. Good packaging. Recommend. Five stars. Happy. Thank you.", 1),
    ("AMAZING PRODUCT BEST IN MARKET VERY FAST DELIVERY SUPERB QUALITY MUST BUY 5 STARS NO COMPLAINTS.", 1),
    ("Product is ok good nice fine alright decent acceptable satisfactory adequate sufficient fine good ok.", 1),

    # More genuine
    ("Used this blender for making smoothies every morning for the past month. It handles ice and frozen fruit without any trouble. The noise level is moderate – not too loud. The container is a bit hard to clean around the blades. Overall very happy.", 0),
    ("The shoes run slightly large so I recommend ordering half size down. The material breathes well during workouts but the cushioning is not adequate for long runs over 5km. Good for casual gym use. The color looks exactly like the photo.", 0),
    ("Third time ordering from this brand. Quality has been consistent. This particular item had a small defect on the seam that wasn't visible in photos – reached out to seller and they offered a partial refund without return. Fair resolution.", 0),
    ("Used as a travel adapter for my Europe trip. Worked in Germany, France, and Italy without issues. Compact size fits easily in the bag. Gets slightly warm when charging multiple devices simultaneously. Would have liked a carrying pouch.", 0),

    # More fake
    ("5 star five star 5/5 excellent excellent excellent excellent best product best quality love it love it amazing amazing perfect perfect.", 1),
    ("RECEIVED TODAY. PACKAGING SUPERB. PRODUCT SUPERB. QUALITY SUPERB. DELIVERY SUPERB. EVERYTHING SUPERB. HIGHLY RECOMMEND. BUY BUY BUY.", 1),
    ("Value for money. Good product. Nice quality. Fast delivery. Happy. Recommend. Five stars. Thank you. Buy it.", 1),
    ("Best ever best ever best ever this product is the best ever I love it so much best ever in my life.", 1),
]

TEXTS  = [d[0] for d in TRAINING_DATA]
LABELS = [d[1] for d in TRAINING_DATA]


def preprocess(text):
    """Light NLP preprocessing."""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline():
    """TF-IDF + Calibrated LinearSVC ensemble."""
    tfidf = TfidfVectorizer(
        preprocessor=preprocess,
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b"
    )
    svc = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3)
    pipeline = Pipeline([("tfidf", tfidf), ("clf", svc)])
    return pipeline


def train_and_save():
    X = TEXTS
    y = LABELS

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("ReviewGuard Model Training")
    print("="*50)
    print(f"\nTraining samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"\nTest Accuracy    : {acc*100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine","Suspicious"]))

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(5, len(X)//10 + 1), shuffle=True, random_state=42)
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # Retrain on full data
    model.fit(X, y)

    # Save model + metadata
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "accuracy": round(acc, 4),
        "cv_accuracy": round(scores.mean(), 4),
        "training_samples": len(X),
        "features": "TF-IDF (1-3 grams) + LinearSVC",
        "classes": ["Genuine", "Suspicious"]
    }
    with open(os.path.join(os.path.dirname(__file__), "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Model saved to {model_path}")
    print("   Run `python app.py` to start the API server.\n")
    return model


if __name__ == "__main__":
    train_and_save()
