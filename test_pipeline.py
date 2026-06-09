import os
import pandas as pd
from src.ingestion import EvidenceIngestion
from src.email_analysis import EmailAnalyzer
from src.behavior_analysis import BehaviorAnalyzer
from src.fusion import FusionEngine
from src.reporting import ReportManager

def test_full_pipeline():
    print("=== Testing Full Pipeline ===")
    
    # 1. Ingestion
    print("\n1. Ingestion Layer")
    ingestion = EvidenceIngestion()
    emails_df = ingestion.load_emails("data/sample_emails.csv")
    logs_df = ingestion.load_logs("data/sample_logs.csv")
    
    if emails_df is None or logs_df is None:
        print("Ingestion failed.")
        return

    # 2. Analysis
    print("\n2. Analysis Layer")
    
    # Email
    print("Running Email Analysis...")
    email_analyzer = EmailAnalyzer()
    email_results = email_analyzer.analyze_emails(emails_df)
    print(f"Analyzed {len(email_results)} emails.")
    
    # Behavior
    print("Running Behavior Analysis...")
    behavior_analyzer = BehaviorAnalyzer()
    if not os.path.exists("models/behavior_model/ae.pth"):
        print("Training behavior models...")
        behavior_analyzer.train(logs_df)
    
    behavior_results = behavior_analyzer.analyze_behavior(logs_df)
    print(f"Detected {len(behavior_results)} behavioral anomalies.")

    # 3. Fusion
    print("\n3. Fusion Layer")
    fusion_engine = FusionEngine()
    signatures = fusion_engine.fuse_evidence(email_results, behavior_results)
    print(f"Generated {len(signatures)} signatures.")
    if not signatures.empty:
        print(signatures[['Crime_Type', 'Severity', 'Confidence_Score', 'Explanation']])
    
    # 4. Reporting
    print("\n4. Reporting Layer")
    report_manager = ReportManager()
    email_stats = {'suspicious_count': len(email_results[email_results['Is_Suspicious']])}
    behavior_stats = {'anomaly_count': len(behavior_results)}
    report_path = report_manager.generate_report(signatures, email_stats, behavior_stats)
    print(f"Report generated at: {report_path}")
    
    print("\n=== Pipeline Test Passed ===")

if __name__ == "__main__":
    test_full_pipeline()
