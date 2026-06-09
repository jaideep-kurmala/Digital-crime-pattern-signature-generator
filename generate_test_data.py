import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_test_data(num_emails=50, num_logs=200):
    print("Generating synthetic test data...")
    
    # --- Configuration ---
    users = [f"user{i:03d}" for i in range(1, 11)] # user001 to user010
    internal_domain = "company.com"
    external_domains = ["gmail.com", "yahoo.com", "partner.com", "malicious-site.net", "fake-support.org"]
    
    base_time = datetime(2023, 11, 1, 8, 0, 0)
    
    # --- 1. Generate Emails ---
    emails = []
    
    # Templates
    benign_subjects = ["Meeting Notes", "Project Update", "Lunch?", "Weekly Report", "Invoice #1234", "Holiday Schedule"]
    benign_bodies = ["Please find attached.", "Let's meet at 2pm.", "Can you review this?", "Thanks for the update.", "See you there."]
    
    phishing_subjects = ["URGENT: Account Suspended", "Verify your password", "You won a prize!", "Immediate Action Required", "Security Alert"]
    phishing_bodies = ["Click here to verify your account.", "Your account will be deleted in 24 hours.", "Claim your iPhone now.", "Suspicious activity detected.", "Please update your payment info."]
    
    for i in range(num_emails):
        timestamp = base_time + timedelta(minutes=random.randint(0, 60*8)) # Within 8 hours
        
        is_phishing = random.random() < 0.2 # 20% Phishing
        
        if is_phishing:
            subject = random.choice(phishing_subjects)
            body = random.choice(phishing_bodies)
            sender_domain = random.choice(external_domains[3:]) # Malicious domains
            sender = f"admin@{sender_domain}"
            receiver = f"{random.choice(users)}@{internal_domain}"
            # Add some targeted attacks
            if i % 5 == 0:
                 sender = "security@company-support.net" # Impersonation
        else:
            subject = random.choice(benign_subjects)
            body = random.choice(benign_bodies)
            sender_domain = random.choice(external_domains[:3] + [internal_domain])
            sender = f"colleague@{sender_domain}"
            receiver = f"{random.choice(users)}@{internal_domain}"
            
        emails.append([subject, body, sender, receiver, timestamp])
    
    emails_df = pd.DataFrame(emails, columns=['Subject', 'Body', 'Sender', 'Receiver', 'Timestamp'])
    emails_df.to_csv("data/test_emails.csv", index=False)
    print(f"Generated {len(emails_df)} emails to data/test_emails.csv")

    # --- 2. Generate Logs ---
    logs = []
    actions = ["LOGIN", "LOGOUT", "ACCESS", "DOWNLOAD", "UPLOAD", "DELETE"]
    resources = ["portal.company.com", "/hr/payroll", "/admin/settings", "salary_data.csv", "project_specs.pdf", "audit_logs"]
    
    # Normal behavior loop
    for user in users:
        # Each user does 10-20 actions
        num_actions = random.randint(10, 20)
        current_time = base_time + timedelta(minutes=random.randint(0, 30)) # Start randomly
        
        for _ in range(num_actions):
            action = random.choice(actions[:3]) # Mostly normal actions
            resource = random.choice(resources[:2])
            logs.append([user, action, resource, current_time])
            current_time += timedelta(minutes=random.randint(1, 15))

    # Anomalous behavior (Coordinated with Phishing)
    # Pick a victim from emails who received phishing
    phishing_victims = emails_df[emails_df['Subject'].isin(phishing_subjects)]['Receiver'].unique()
    if len(phishing_victims) > 0:
        victim_email = random.choice(phishing_victims)
        victim_user = victim_email.split('@')[0]
        
        print(f"Injecting attack scenario for {victim_user}...")
        
        # Find time of phishing email
        attack_time = emails_df[emails_df['Receiver'] == victim_email]['Timestamp'].iloc[0] + timedelta(minutes=10)
        
        # Malicious Sequence
        malicious_sequence = [
            ("LOGIN", "portal.company.com"),
            ("ACCESS", "/admin/settings"),
            ("DOWNLOAD", "salary_data.csv"),
            ("DELETE", "audit_logs"),
            ("UPLOAD", "backdoor.exe")
        ]
        
        for action, resource in malicious_sequence:
            logs.append([victim_user, action, resource, attack_time])
            attack_time += timedelta(minutes=2)

    logs_df = pd.DataFrame(logs, columns=['User ID', 'Action', 'Resource', 'Timestamp'])
    # Sort by time
    logs_df.sort_values(by='Timestamp', inplace=True)
    logs_df.to_csv("data/test_logs.csv", index=False)
    print(f"Generated {len(logs_df)} logs to data/test_logs.csv")

if __name__ == "__main__":
    generate_test_data()
