"""
ReviewGuard – Model Trainer
Trains a TF-IDF + Classifier pipeline on labeled review data.
Run this once before starting the API server: python model_trainer.py
"""

import pickle, os, json, csv
import numpy as np # pyright: ignore[reportUnusedImport]
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB # type: ignore
from sklearn.linear_model import LogisticRegression # pyright: ignore[reportUnusedImport]
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.calibration import CalibratedClassifierCV

from backend.app import preprocess



def load_csv_data(csv_path): # type: ignore
    """Load training data from CSV file."""
    texts, labels = [], []
    if not os.path.exists(csv_path): # type: ignore
        print(f"CSV file not found: {csv_path}")
        return texts, labels # type: ignore
    
    with open(csv_path, "r", encoding="utf-8-sig") as f: # type: ignore
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text_", "").strip()
            label = row.get("label", "").strip()
            if text and label and label in ("0", "1"):
                texts.append(text) # type: ignore
                labels.append(int(label)) # type: ignore
    
    print(f"Loaded {len(texts)} reviews from CSV ({sum(labels)} fake, {len(labels)-sum(labels)} genuine)") # type: ignore
    return texts, labels # type: ignore




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
    csv_path = os.path.join(os.path.dirname(__file__), "fake reviews dataset.csv")
    X_csv, y_csv = load_csv_data(csv_path)
    
    X, y = X_csv, y_csv
    print(f"Using {len(X)} samples from CSV dataset") # type: ignore

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split( # type: ignore
        X, y, test_size=0.2, random_state=42, stratify=y # type: ignore
    )

    model = build_pipeline()
    model.fit(X_train, y_train) # type: ignore

    y_pred = model.predict(X_test) # type: ignore
    acc = accuracy_score(y_test, y_pred) # type: ignore

    print("\n" + "="*50)
    print("ReviewGuard Model Training")
    print("="*50)
    print(f"\nTraining samples : {len(X_train)}") # type: ignore
    print(f"Test samples     : {len(X_test)}") # type: ignore
    print(f"\nTest Accuracy    : {acc*100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine","Suspicious"])) # type: ignore

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(5, len(X)//10 + 1), shuffle=True, random_state=42) # type: ignore
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="accuracy") # type: ignore
    print(f"CV Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # Retrain on full data
    model.fit(X, y) # type: ignore

    # Save model + metadata
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = { # type: ignore
        "accuracy": round(acc, 4),
        "cv_accuracy": round(scores.mean(), 4),
        "training_samples": len(X), # type: ignore
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
