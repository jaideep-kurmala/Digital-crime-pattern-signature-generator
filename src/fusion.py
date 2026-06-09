import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# Rule Definitions
# ──────────────────────────────────────────────
RULE_DEFINITIONS = {
    "PHISHING_HIGH_CONFIDENCE": {
        "description": "Phishing probability exceeds 0.7 confidence threshold",
        "weight": 0.35,
    },
    "MULTI_EMAIL_CAMPAIGN": {
        "description": "Multiple suspicious emails detected indicating coordinated campaign",
        "weight": 0.20,
    },
    "USER_BEHAVIOR_DRIFT": {
        "description": "User behaviour anomaly score exceeds normal threshold",
        "weight": 0.25,
    },
    "EMAIL_BEHAVIOR_CORRELATION": {
        "description": "Email threat and behavioural anomaly correlated within time window",
        "weight": 0.30,
    },
    "MULTI_USER_AFFECTED": {
        "description": "Multiple distinct users impacted by related threat indicators",
        "weight": 0.15,
    },
}

SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class FusionEngine:
    """
    Enhanced Fusion Engine with:
      • Explicit triggered-rules list
      • Rule-based severity escalation (LOW → MEDIUM → HIGH → CRITICAL)
      • Severity-escalation-steps audit trail
    """

    def __init__(self):
        self.time_window_hours = 24
        self.weights = {"email": 0.4, "behavior": 0.6}

    # ================================================================
    # Public API
    # ================================================================
    def fuse_evidence(self, email_results, behavior_results):
        """
        Fuses email and behaviour analysis results.
        Returns a DataFrame whose every row contains:
          - Crime_Type, Confidence_Score, Explanation
          - triggered_rules          (list[dict])
          - severity_escalation_steps (list[dict])
          - Severity                 (final after escalation)
          … plus legacy fields kept for compatibility.
        """
        signatures = []

        # ----- pre-processing -----
        suspicious_emails = email_results[email_results["Is_Suspicious"] == True].copy()
        if not behavior_results.empty:
            anomalous_behavior = behavior_results.copy()
        else:
            anomalous_behavior = pd.DataFrame(
                columns=["User ID", "Sequence_Start_Time", "Anomaly_Score", "Details"]
            )

        # Track affected users for MULTI_USER_AFFECTED rule
        affected_users = set()

        # --- collect per-email items for campaign rule ---
        suspicious_email_count = len(suspicious_emails)

        # ----- 1. Correlated (email + behaviour) signatures -----
        for _, email in suspicious_emails.iterrows():
            related_behavior = pd.DataFrame()
            if not anomalous_behavior.empty:
                email_time = email["Timestamp"]
                start_time = email_time - pd.Timedelta(hours=1)
                end_time = email_time + pd.Timedelta(hours=self.time_window_hours)
                related_behavior = anomalous_behavior[
                    (anomalous_behavior["Sequence_Start_Time"] >= start_time)
                    & (anomalous_behavior["Sequence_Start_Time"] <= end_time)
                ]

            if not related_behavior.empty:
                behavior_row = related_behavior.iloc[0]
                sig = self._create_signature(
                    sig_type="Coordinated Phishing & Compromise",
                    email_data=email,
                    behavior_data=behavior_row,
                    confidence=0.95,
                    suspicious_email_count=suspicious_email_count,
                    affected_users=affected_users,
                )
                signatures.append(sig)
            else:
                sig = self._create_signature(
                    sig_type="Phishing Campaign",
                    email_data=email,
                    behavior_data=None,
                    confidence=float(email["Phishing_Probability"]),
                    suspicious_email_count=suspicious_email_count,
                    affected_users=affected_users,
                )
                signatures.append(sig)

        # ----- 2. Isolated behaviour anomalies -----
        if not anomalous_behavior.empty:
            for _, behavior in anomalous_behavior.iterrows():
                sig = self._create_signature(
                    sig_type="Insider Threat / Account Compromise",
                    email_data=None,
                    behavior_data=behavior,
                    confidence=min(behavior["Anomaly_Score"] / 2.0, 0.99),
                    suspicious_email_count=suspicious_email_count,
                    affected_users=affected_users,
                )
                signatures.append(sig)

        return pd.DataFrame(signatures)

    # ================================================================
    # Internal helpers
    # ================================================================
    def _escalate_severity(self, triggered_rules):
        """
        Walk the triggered rules and escalate severity one level per rule.
        Returns (final_severity, escalation_steps).
        """
        level = 0  # starts at LOW
        steps = []
        step_number = 0

        for rule in triggered_rules:
            rule_id = rule["rule_id"]
            prev = SEVERITY_LEVELS[level]
            if level < len(SEVERITY_LEVELS) - 1:
                level += 1
            new = SEVERITY_LEVELS[level]
            step_number += 1
            steps.append(
                {
                    "step_number": step_number,
                    "rule_triggered": rule_id,
                    "previous_severity": prev,
                    "new_severity": new,
                }
            )
        return SEVERITY_LEVELS[level], steps

    def _collect_triggered_rules(
        self,
        email_data,
        behavior_data,
        confidence,
        suspicious_email_count,
        affected_users,
    ):
        """
        Evaluate every rule and return only those that fire.
        Each entry: {rule_id, rule_description, evidence_source, rule_weight}
        """
        rules = []

        # ---- PHISHING_HIGH_CONFIDENCE ----
        if email_data is not None:
            phish_prob = float(email_data["Phishing_Probability"])
            if phish_prob > 0.7:
                rules.append(
                    {
                        "rule_id": "PHISHING_HIGH_CONFIDENCE",
                        "rule_description": RULE_DEFINITIONS["PHISHING_HIGH_CONFIDENCE"]["description"],
                        "evidence_source": f"Email '{email_data['Subject']}' (prob={phish_prob:.2f})",
                        "rule_weight": RULE_DEFINITIONS["PHISHING_HIGH_CONFIDENCE"]["weight"],
                    }
                )

        # ---- MULTI_EMAIL_CAMPAIGN ----
        if suspicious_email_count >= 3:
            rules.append(
                {
                    "rule_id": "MULTI_EMAIL_CAMPAIGN",
                    "rule_description": RULE_DEFINITIONS["MULTI_EMAIL_CAMPAIGN"]["description"],
                    "evidence_source": f"{suspicious_email_count} suspicious emails in batch",
                    "rule_weight": RULE_DEFINITIONS["MULTI_EMAIL_CAMPAIGN"]["weight"],
                }
            )

        # ---- USER_BEHAVIOR_DRIFT ----
        if behavior_data is not None:
            anomaly_threshold = 1.5  # same threshold used in behavior_analysis
            if behavior_data["Anomaly_Score"] > anomaly_threshold:
                rules.append(
                    {
                        "rule_id": "USER_BEHAVIOR_DRIFT",
                        "rule_description": RULE_DEFINITIONS["USER_BEHAVIOR_DRIFT"]["description"],
                        "evidence_source": f"User {behavior_data['User ID']} (anomaly={behavior_data['Anomaly_Score']:.2f})",
                        "rule_weight": RULE_DEFINITIONS["USER_BEHAVIOR_DRIFT"]["weight"],
                    }
                )

        # ---- EMAIL_BEHAVIOR_CORRELATION ----
        if email_data is not None and behavior_data is not None:
            rules.append(
                {
                    "rule_id": "EMAIL_BEHAVIOR_CORRELATION",
                    "rule_description": RULE_DEFINITIONS["EMAIL_BEHAVIOR_CORRELATION"]["description"],
                    "evidence_source": "Email threat + behaviour anomaly within time window",
                    "rule_weight": RULE_DEFINITIONS["EMAIL_BEHAVIOR_CORRELATION"]["weight"],
                }
            )

        # ---- MULTI_USER_AFFECTED ----
        if len(affected_users) >= 2:
            rules.append(
                {
                    "rule_id": "MULTI_USER_AFFECTED",
                    "rule_description": RULE_DEFINITIONS["MULTI_USER_AFFECTED"]["description"],
                    "evidence_source": f"{len(affected_users)} distinct users affected",
                    "rule_weight": RULE_DEFINITIONS["MULTI_USER_AFFECTED"]["weight"],
                }
            )

        return rules

    def _create_signature(
        self,
        sig_type,
        email_data,
        behavior_data,
        confidence,
        suspicious_email_count,
        affected_users,
    ):
        """Build one crime-pattern signature dict with all new fields."""

        # --- Track affected users ---
        if behavior_data is not None:
            affected_users.add(behavior_data["User ID"])
        if email_data is not None:
            affected_users.add(email_data["Sender"])

        # --- Triggered Rules ---
        triggered_rules = self._collect_triggered_rules(
            email_data, behavior_data, confidence, suspicious_email_count, affected_users
        )

        # --- Severity Escalation ---
        severity, escalation_steps = self._escalate_severity(triggered_rules)

        # --- Legacy explanation (still kept for backward compat) ---
        explanation_parts = []
        email_contrib = 0
        behavior_contrib = 0

        if email_data is not None:
            explanation_parts.append(
                f"Detected Suspicious Email: '{email_data['Subject']}' from {email_data['Sender']}."
            )
            if email_data["Explainability"]:
                explanation_parts.append(
                    f"Email Indicators: {', '.join(email_data['Explainability'])}"
                )

        if behavior_data is not None:
            explanation_parts.append(
                f"Detected Anomalous Behavior for User {behavior_data['User ID']}."
            )
            explanation_parts.append(f"Behavior Indicators: {behavior_data['Details']}")

        if email_data is not None and behavior_data is not None:
            email_contrib, behavior_contrib = 40, 60
        elif email_data is not None:
            email_contrib = 100
        elif behavior_data is not None:
            behavior_contrib = 100

        # --- Determine involved user & timestamp ---
        involved_user = (
            behavior_data["User ID"]
            if behavior_data is not None
            else (email_data["Sender"] if email_data is not None else "Unknown")
        )
        timestamp = (
            behavior_data["Sequence_Start_Time"]
            if behavior_data is not None
            else (email_data["Timestamp"] if email_data is not None else pd.Timestamp.now())
        )

        # --- Determine event_type tag for timeline ---
        if email_data is not None and behavior_data is not None:
            event_type = "Fusion event"
        elif email_data is not None:
            event_type = "Email threat"
        else:
            event_type = "Behavior anomaly"

        return {
            "Crime_Type": sig_type,
            "Severity": severity,
            "Confidence_Score": float(confidence),
            "Explanation": " | ".join(explanation_parts),
            "Involved_User": involved_user,
            "Timestamp": timestamp,
            "Email_Contribution_Pct": email_contrib,
            "Behavior_Contribution_Pct": behavior_contrib,
            # === NEW structured fields ===
            "triggered_rules": triggered_rules,
            "severity_escalation_steps": escalation_steps,
            "event_type": event_type,
        }


if __name__ == "__main__":
    # Test stub
    pass
