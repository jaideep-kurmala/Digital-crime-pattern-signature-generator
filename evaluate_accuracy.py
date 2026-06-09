import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.ingestion import EvidenceIngestion
from src.email_analysis import EmailAnalyzer
from src.behavior_analysis import BehaviorAnalyzer

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------
# Email evaluation
# ------------------

def evaluate_email():
    ingestion = EvidenceIngestion()
    emails = ingestion.load_emails("data/sample_emails.csv")
    if emails is None:
        raise RuntimeError("Failed to load email sample")

    # This script assumes sample_emails.csv has a column `Is_Phishing` as ground truth.
    if "Is_Phishing" not in emails.columns:
        print("WARNING: No ground truth `Is_Phishing` column found in data/sample_emails.csv")
        print("Adding dummy labels (0) for demo.")
        emails["Is_Phishing"] = 0

    analyzer = EmailAnalyzer()
    results = analyzer.analyze_emails(emails)

    y_true = emails["Is_Phishing"].astype(int)
    y_score = results["Phishing_Probability"]
    y_pred = (y_score > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # roc_auc_score only if both classes present
    if len(y_true.unique()) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = None

    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "email_accuracy.csv"), index=False)
    print("Email accuracy saved to output/email_accuracy.csv")


# ------------------
# Behavior evaluation (demo)
# ------------------

def evaluate_behavior():
    ingestion = EvidenceIngestion()
    logs = ingestion.load_logs("data/sample_logs.csv")

    if logs is None:
        raise RuntimeError("Failed to load log sample")

    analyzer = BehaviorAnalyzer()
    # Ensure model available; train if needed
    if not analyzer.load_models():
        analyzer.train(logs)

    behavior_results = analyzer.analyze_behavior(logs)

    # Handle no anomalies found
    if behavior_results.empty:
        print("No behavior anomalies detected; creating empty summary.")
        pd.DataFrame(columns=["User ID", "Sequence_Start_Time", "Anomaly_Score", "AE_Error", "LSTM_Deviation", "Details"]).to_csv(
            os.path.join(OUTPUT_DIR, "behavior_top_anomalies.csv"), index=False
        )
    else:
        output_path = os.path.join(OUTPUT_DIR, "behavior_top_anomalies.csv")
        behavior_results.sort_values("Anomaly_Score", ascending=False).head(20).to_csv(output_path, index=False)
        print(f"Top behavior anomalies saved to {output_path}")

    # If ground truth column exists, compute additional metrics
    if "is_anomaly" in logs.columns:
        merged = behavior_results.merge(logs, left_on=["User ID", "Sequence_Start_Time"], right_on=["User ID", "Timestamp"], how="left")
        y_true = merged["is_anomaly"].fillna(0).astype(int)
        y_pred = (merged["Anomaly_Score"] > 1.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "behavior_accuracy.csv"), index=False)
        print("Behavior accuracy saved to output/behavior_accuracy.csv")
    else:
        print("No ground truth anomaly labels found in logs; behavior accuracy not computed.")


if __name__ == "__main__":
    evaluate_email()
    evaluate_behavior()
