import pandas as pd
import numpy as np

class EvidenceIngestion:
    def __init__(self):
        self.email_schema = ['Subject', 'Body', 'Sender', 'Receiver', 'Timestamp']
        self.log_schema = ['User ID', 'Action', 'Resource', 'Timestamp']

    def load_emails(self, file_path):
        """
        Loads and cleans email evidence CSV.
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate Schema
            if not all(col in df.columns for col in self.email_schema):
                missing = [col for col in self.email_schema if col not in df.columns]
                raise ValueError(f"Missing columns in Email CSV: {missing}")
            
            # Select relevant columns
            df = df[self.email_schema].copy()
            
            # Remove missing values
            df.dropna(inplace=True)
            
            # Normalize Timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
            
            # Text Normalization (basic)
            df['Subject'] = df['Subject'].astype(str).str.strip()
            df['Body'] = df['Body'].astype(str).str.strip()
            df['Sender'] = df['Sender'].astype(str).str.strip().str.lower()
            df['Receiver'] = df['Receiver'].astype(str).str.strip().str.lower()
            
            print(f"Loaded {len(df)} valid email records.")
            return df
            
        except Exception as e:
            print(f"Error loading emails: {e}")
            return None

    def load_logs(self, file_path):
        """
        Loads and cleans user activity logs CSV.
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate Schema
            if not all(col in df.columns for col in self.log_schema):
                missing = [col for col in self.log_schema if col not in df.columns]
                raise ValueError(f"Missing columns in Log CSV: {missing}")
            
            # Select relevant columns
            df = df[self.log_schema].copy()
            
            # Remove missing values
            df.dropna(inplace=True)
            
            # Normalize Timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
            
            # String Normalization
            df['User ID'] = df['User ID'].astype(str).str.strip()
            df['Action'] = df['Action'].astype(str).str.strip().str.upper()
            df['Resource'] = df['Resource'].astype(str).str.strip()
            
            # Sort by Timestamp
            df.sort_values(by='Timestamp', inplace=True)
            
            print(f"Loaded {len(df)} valid log records.")
            return df
            
        except Exception as e:
            print(f"Error loading logs: {e}")
            return None

if __name__ == "__main__":
    ingestion = EvidenceIngestion()
    emails = ingestion.load_emails("data/sample_emails.csv")
    logs = ingestion.load_logs("data/sample_logs.csv")
    
    if emails is not None:
        print(emails.head())
    if logs is not None:
        print(logs.head())
