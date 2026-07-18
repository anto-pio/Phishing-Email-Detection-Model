import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def generate_mock_data():
    """Generates a robust mock dataset representing real-world phishing vs safe emails."""
    data = {
        "email_text": [
            "Dear valued customer, your bank account has been suspended due to suspicious activity. Click http://secure-login-update-bank.com immediately to verify your identity.",
            "Hey, are we still meeting up for lunch today at the downtown deli? Let me know when you leave the office.",
            "URGENT: Your Netflix subscription failed to renew. Update your credit card details immediately at http://netflx-billing-portal.net/login or face cancellation.",
            "Attached is the monthly financial report for Q3. Please review the spreadsheet updates before our alignment meeting tomorrow morning.",
            "Congratulations! You have won a $1000 Amazon Gift Card! Click here http://win-free-rewards.xyz to claim your lottery prize now.",
            "Hi team, just a reminder that the performance review self-evaluations are due by Friday afternoon. Let HR know if you hit any roadblocks.",
            "Official security notice from IT: Your password expires in 24 hours. Reset it now on the corporate dashboard link http://internal-update-auth.com.",
            "Thanks for registering for the upcoming Python developer conference. Your ticket receipt and schedule breakdown are attached below.",
        ],
        # 1 = Phishing, 0 = Safe
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def engineer_features(df):
    """Computes explicit structural metadata features out of the raw email strings."""
    # Count occurrences of standard URL indicators
    df["url_count"] = df["email_text"].apply(
        lambda x: x.count("http://") + x.count("https://") + x.count("www.")
    )

    # Check for heavy manipulative urgency keywords
    urgency_words = ["urgent", "suspended", "immediately", "congratulations", "won"]
    df["urgency_score"] = df["email_text"].apply(
        lambda x: sum(1 for word in urgency_words if word in x.lower())
    )

    return df


def build_and_evaluate_model():
    # 1. Load Data
    df = generate_mock_data()

    # 2. Feature Engineering
    df = engineer_features(df)

    # 3. Text Vectorization (TF-IDF converts words to numerical importance scores)
    # Using stop_words='english' filters out non-contextual noise like 'the', 'is', 'at'
    tfidf = TfidfVectorizer(stop_words="english", max_features=100)
    tfidf_matrices = tfidf.fit_transform(df["email_text"]).toarray()

    # Combine TF-IDF text arrays with our engineered meta-features
    meta_features = df[["url_count", "urgency_score"]].values
    X = np.hstack((tfidf_matrices, meta_features))
    y = df["label"].values

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5. Model Initialization and Training
    # Logistic Regression is highly effective for high-dimensional text matrices
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("=== MODEL PERFORMANCE EVALUATION ===")
    print(f"Overall Model Accuracy: {accuracy * 100:.2f}%\n")

    print("--- Classification Report ---")
    print(
        classification_report(
            y_test, predictions, target_names=["Safe (0)", "Phishing (1)"]
        )
    )

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Safe", "Actual Phishing"],
        columns=["Predicted Safe", "Predicted Phishing"],
    )
    print(cm_df)

    return model, tfidf


if __name__ == "__main__":
    trained_model, vectorizer = build_and_evaluate_model()