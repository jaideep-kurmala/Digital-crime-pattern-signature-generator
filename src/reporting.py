from fpdf import FPDF
import pandas as pd
import os


class ForensicReportGenerator(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Digital Crime Pattern Signature Report", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, "L", 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, str(body))
        self.ln()

    def add_table(self, df):
        self.set_font("Arial", "B", 10)
        for col in df.columns:
            self.cell(40, 10, str(col)[:15], 1)
        self.ln()
        self.set_font("Arial", "", 9)
        for _, row in df.iterrows():
            for col in df.columns:
                self.cell(40, 10, str(row[col])[:15], 1)
            self.ln()
        self.ln()


class ReportManager:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir

    def generate_report(self, signatures_df, email_stats, behavior_stats):
        pdf = ForensicReportGenerator()
        pdf.add_page()
        epw = pdf.w - 2 * pdf.l_margin  # effective page width

        # ──────────────────────────────────────────
        # 1. Executive Summary
        # ──────────────────────────────────────────
        pdf.chapter_title("1. Executive Summary")
        summary_text = (
            f"Total Suspicious Emails Detected: {email_stats['suspicious_count']}\n"
            f"Total Behavioral Anomalies: {behavior_stats['anomaly_count']}\n"
            f"Critical Crime Signatures: "
            f"{len(signatures_df[signatures_df['Severity'] == 'CRITICAL']) if not signatures_df.empty else 0}\n"
        )
        pdf.chapter_body(summary_text)

        # ──────────────────────────────────────────
        # 2. Crime Signatures & Explainability
        # ──────────────────────────────────────────
        pdf.chapter_title("2. Digital Crime Signatures & Explainability")
        if not signatures_df.empty:
            for i, row in signatures_df.iterrows():
                pdf.set_font("Arial", "B", 11)
                pdf.cell(
                    0,
                    10,
                    f"Signature #{i+1}: {row['Crime_Type']} ({row['Severity']})",
                    0,
                    1,
                )
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(epw, 8, f"Confidence: {row['Confidence_Score']:.2f}")

                explanation = (
                    str(row["Explanation"])
                    .encode("latin-1", "replace")
                    .decode("latin-1")
                )
                pdf.multi_cell(epw, 8, f"Explanation: {explanation}")
                pdf.multi_cell(
                    epw,
                    8,
                    f"Evidence Contribution: Email {row['Email_Contribution_Pct']}%, "
                    f"Behavior {row['Behavior_Contribution_Pct']}%",
                )
                pdf.ln(5)
        else:
            pdf.chapter_body("No significant crime patterns detected.")

        # ──────────────────────────────────────────
        # 3. Severity Justification  (NEW)
        # ──────────────────────────────────────────
        pdf.add_page()
        pdf.chapter_title("3. Severity Justification")
        if not signatures_df.empty:
            for i, row in signatures_df.iterrows():
                steps = row.get("severity_escalation_steps", [])
                if not steps:
                    continue

                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 10, f"Signature #{i+1}: {row['Crime_Type']}", 0, 1)
                pdf.set_font("Arial", "", 10)

                # Starting severity
                starting = steps[0]["previous_severity"] if steps else "LOW"
                pdf.multi_cell(epw, 8, f"Starting Severity: {starting}")

                # Each escalation step
                for step in steps:
                    step_text = (
                        f"  Step {step['step_number']}: "
                        f"{step['rule_triggered']}  |  "
                        f"{step['previous_severity']} -> {step['new_severity']}"
                    )
                    step_text = step_text.encode("latin-1", "replace").decode("latin-1")
                    pdf.multi_cell(epw, 7, step_text)

                # Final severity
                pdf.set_font("Arial", "B", 10)
                pdf.multi_cell(epw, 8, f"Final Severity: {row['Severity']}")
                pdf.ln(4)
        else:
            pdf.chapter_body("No escalation steps to display.")

        # ──────────────────────────────────────────
        # 4. Triggered Fusion Rules  (NEW)
        # ──────────────────────────────────────────
        pdf.chapter_title("4. Triggered Fusion Rules")
        if not signatures_df.empty:
            for i, row in signatures_df.iterrows():
                rules = row.get("triggered_rules", [])
                if not rules:
                    continue

                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 10, f"Signature #{i+1}: {row['Crime_Type']}", 0, 1)

                # Table header
                pdf.set_font("Arial", "B", 9)
                col_widths = [45, 55, 60, 25]  # rule_id, desc, evidence, weight
                headers = ["Rule ID", "Description", "Evidence Source", "Weight"]
                for h, w in zip(headers, col_widths):
                    pdf.cell(w, 8, h, 1)
                pdf.ln()

                # Table rows
                pdf.set_font("Arial", "", 8)
                for rule in rules:
                    vals = [
                        str(rule["rule_id"])[:22],
                        str(rule["rule_description"])[:28],
                        str(rule["evidence_source"])[:30],
                        f"{rule['rule_weight']:.2f}",
                    ]
                    for v, w in zip(vals, col_widths):
                        safe = v.encode("latin-1", "replace").decode("latin-1")
                        pdf.cell(w, 8, safe, 1)
                    pdf.ln()
                pdf.ln(4)
        else:
            pdf.chapter_body("No triggered rules to display.")

        # ──────────────────────────────────────────
        # 5. Forensic Timeline (Enhanced)
        # ──────────────────────────────────────────
        pdf.add_page()
        pdf.chapter_title("5. Forensic Timeline")
        if not signatures_df.empty:
            timeline_df = signatures_df[
                ["Timestamp", "Crime_Type", "Severity", "event_type"]
            ].sort_values(by="Timestamp")

            pdf.set_font("Arial", "B", 10)
            col_w = [45, 30, 60, 30]
            h_labels = ["Timestamp", "Event Type", "Crime Type", "Severity"]
            for lbl, w in zip(h_labels, col_w):
                pdf.cell(w, 10, lbl, 1)
            pdf.ln()

            pdf.set_font("Arial", "", 9)
            for _, trow in timeline_df.iterrows():
                pdf.cell(col_w[0], 10, str(trow["Timestamp"])[:20], 1)
                pdf.cell(col_w[1], 10, str(trow["event_type"])[:18], 1)
                pdf.cell(col_w[2], 10, str(trow["Crime_Type"])[:30], 1)
                pdf.cell(col_w[3], 10, str(trow["Severity"]), 1)
                pdf.ln()
        else:
            pdf.chapter_body("No events to display.")

        # ──────────────────────────────────────────
        # Save
        # ──────────────────────────────────────────
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "Forensic_Report.pdf")
        pdf.output(output_path)
        return output_path
