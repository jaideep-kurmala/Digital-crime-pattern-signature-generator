import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

class EmailAnalyzer:
    def __init__(self, model_path="models/email_model"):
        print(f"Loading Email Analysis Model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def analyze_emails(self, emails_df):
        """
        Analyzes a dataframe of emails for phishing.
        """
        results = []
        
        for index, row in emails_df.iterrows():
            subject = str(row['Subject'])
            body = str(row['Body'])
            text = f"{subject} {body}"
            
            # 1. Model Inference
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).numpy()[0]
                # Assuming index 1 is Phishing
                phishing_prob = probs[1] if len(probs) > 1 else probs[0] 
                
            # 2. Rule Extraction (Explainability)
            rules_triggered = self._extract_rules(text, row['Sender'])
            
            result = {
                'Subject': subject,
                'Sender': row['Sender'],
                'Timestamp': row['Timestamp'],
                'Phishing_Probability': float(phishing_prob),
                'Is_Suspicious': bool(phishing_prob > 0.5), # Threshold
                'Explainability': rules_triggered
            }
            results.append(result)
            
        return pd.DataFrame(results)

    def _extract_rules(self, text, sender):
        """
        Extracts explainable rule indicators using extensive keyword matching.
        Note: Detection is primarily Model-Driven (Transformer). These rules are for Explainability only.
        """
        text_lower = text.lower()
        triggers = []
        
        # 1. Urgency Keywords
        urgency_keywords = [
            'urgent', 'immediate', 'action required', 'verify now', 'account suspended',
            'security alert', 'final warning', 'limited time', 'confirm identity', 'restricted',
            'suspend', 'terminate', 'expire'
        ]
        if any(keyword in text_lower for keyword in urgency_keywords):
            triggers.append("Urgency Detected")
            
        # 2. Financial / Scam Language
        scam_keywords = [
            'password reset', 'login attempt', 'payment failed', 'billing issue', 'update details',
            'bank alert', 'credit card', 'refund', 'transaction failed', 'wallet',
            'lottery', 'win', 'prize', 'verify your account', 'bank account', 'social security'
        ]
        if any(keyword in text_lower for keyword in scam_keywords):
            triggers.append("Financial/Scam Language")

        # 3. Impersonation Indicators (Sender & Content)
        impersonation_keywords = [
            'admin', 'support team', 'it department', 'security team', 'customer service',
            'microsoft', 'paypal', 'amazon', 'bank', 'helpdesk'
        ]
        
        # Check Content for brand names
        if any(keyword in text_lower for keyword in impersonation_keywords):
            triggers.append("Sensitive Brand/Department Mention")
            
        # Check Sender for potential spoofing (Simple check)
        if any(keyword in sender.lower() for keyword in ['admin', 'support', 'security', 'service']):
            if "company.com" not in sender: # Assuming company.com is the internal domain
                triggers.append("Potential Impersonation (External Domain)")
                
        return triggers

if __name__ == "__main__":
    # Test
    from ingestion import EvidenceIngestion
    
    ingestion = EvidenceIngestion()
    emails = ingestion.load_emails("data/sample_emails.csv")
    
    analyzer = EmailAnalyzer()
    results = analyzer.analyze_emails(emails)
    print(results[['Subject', 'Phishing_Probability', 'Is_Suspicious', 'Explainability']])
