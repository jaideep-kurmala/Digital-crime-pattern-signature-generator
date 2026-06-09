import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

class BehaviorPreprocessor:
    def __init__(self):
        self.le = LabelEncoder()
        self.window_size = 3 # Small window for demo
        self.vocab_size = 0

    def fit_transform(self, df):
        # Combine Action and Resource to create unique events
        df['Event'] = df['Action'] + "_" + df['Resource']
        self.le.fit(df['Event'])
        self.vocab_size = len(self.le.classes_)
        encoded = self.le.transform(df['Event'])
        return self._create_windows(encoded)

    def transform(self, df):
        # Handle unseen labels by mapping to a generic 'unknown' or handling exception
        # For simplicity, we assume consistent vocabulary or ignore unknowns
        df['Event'] = df['Action'] + "_" + df['Resource']
        # Filter unknown
        known_mask = df['Event'].isin(self.le.classes_)
        if not known_mask.all():
            print(f"Warning: Dropping { (~known_mask).sum() } unknown events")
            df = df[known_mask]
        
        encoded = self.le.transform(df['Event'])
        return self._create_windows(encoded)

    def _create_windows(self, sequence):
        windows = []
        if len(sequence) < self.window_size:
            # Pad if too short
            return np.array([np.pad(sequence, (0, self.window_size - len(sequence)), 'constant')])
            
        for i in range(len(sequence) - self.window_size + 1):
            windows.append(sequence[i:i+self.window_size])
        return np.array(windows)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        # Predict based on last hidden state
        logits = self.fc(lstm_out[:, -1, :]) 
        return logits

class BehaviorAnalyzer:
    def __init__(self, model_dir="models/behavior_model"):
        self.model_dir = model_dir
        self.preprocessor = BehaviorPreprocessor()
        self.ae = None
        self.lstm = None
        
        # Paths
        self.ae_path = os.path.join(model_dir, "ae.pth")
        self.lstm_path = os.path.join(model_dir, "lstm.pth")
        self.prep_path = os.path.join(model_dir, "preprocessor.pkl")

    def train(self, logs_df):
        print("Training Behavior Models...")
        windows = self.preprocessor.fit_transform(logs_df)
        
        # Save preprocessor
        with open(self.prep_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)

        # Convert to Tensor
        # For AE, we treat window as a flat feature vector (one-hot approximation or just embeddings)
        # Using simple normalized integer input for AE is suboptimal, usually we use one-hot.
        # Let's use FloatTensor of normalized indices for simplicity in this demo structure
        X_train = torch.FloatTensor(windows) / self.preprocessor.vocab_size 
        X_train_lstm = torch.LongTensor(windows)

        # Train AE
        input_dim = self.preprocessor.window_size
        self.ae = Autoencoder(input_dim)
        optimizer_ae = optim.Adam(self.ae.parameters(), lr=0.01)
        criterion_ae = nn.MSELoss()
        
        for epoch in range(50): # 50 Epochs
            optimizer_ae.zero_grad()
            outputs = self.ae(X_train)
            loss = criterion_ae(outputs, X_train)
            loss.backward()
            optimizer_ae.step()
        
        torch.save(self.ae.state_dict(), self.ae_path)
        print("Autoencoder trained and saved.")

        # Train LSTM (Next Token Prediction style or Reconstruction)
        # Here we train to predict the *last* item in window from previous items?
        # Or just unsupervised reconstruction of the sequence?
        # Spec says "Learns action sequences". Let's do simple next-token prediction
        # But our windows are fixed. Let's just run LSTM on the window and try to reconstruct input (Autoencoder style) or predict next.
        # For "Behavioral Drift", Sequence Likelihood is best.
        # We'll use the LSTM to predict the next token given context.
        # But to keep it simple with the windows:
        # We will train LSTM to predict the sequence itself (Teacher Forcing) or just learn the distribution.
        
        # Let's stick to spec: "Detects behavioral drift".
        # We'll train it to predict the *last* event in the window given the first N-1.
        
        if self.preprocessor.window_size > 1:
            X_lstm_in = X_train_lstm[:, :-1]
            y_lstm_target = X_train_lstm[:, -1]
            
            self.lstm = LSTMModel(self.preprocessor.vocab_size)
            optimizer_lstm = optim.Adam(self.lstm.parameters(), lr=0.01)
            criterion_lstm = nn.CrossEntropyLoss()
            
            for epoch in range(50):
                optimizer_lstm.zero_grad()
                outputs = self.lstm(X_lstm_in)
                loss = criterion_lstm(outputs, y_lstm_target)
                loss.backward()
                optimizer_lstm.step()
                
            torch.save(self.lstm.state_dict(), self.lstm_path)
            print("LSTM trained and saved.")
        else:
            print("Window size too small for LSTM.")

    def load_models(self):
        try:
            with open(self.prep_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            self.ae = Autoencoder(self.preprocessor.window_size)
            self.ae.load_state_dict(torch.load(self.ae_path))
            self.ae.eval()
            
            if os.path.exists(self.lstm_path):
                self.lstm = LSTMModel(self.preprocessor.vocab_size)
                self.lstm.load_state_dict(torch.load(self.lstm_path))
                self.lstm.eval()
                
            print("Models loaded successfully.")
            return True
        except (FileNotFoundError, AttributeError, EOFError) as e:
            print(f"Error loading models ({e}). Will re-train.")
            return False

    def analyze_behavior(self, logs_df):
        if not self.ae:
            if not self.load_models():
                # If loading fails, train on the fly (for demo purposes)
                self.train(logs_df)
        
        # Prepare Data
        # Group by User
        results = []
        users = logs_df['User ID'].unique()
        
        for user in users:
            user_logs = logs_df[logs_df['User ID'] == user].copy()
            if len(user_logs) < self.preprocessor.window_size:
                continue # Not enough data
                
            windows = self.preprocessor.transform(user_logs)
            if len(windows) == 0: continue

            # AE Score
            X_ae = torch.FloatTensor(windows) / self.preprocessor.vocab_size
            with torch.no_grad():
                reconstructed = self.ae(X_ae)
                ae_loss = torch.mean((X_ae - reconstructed) ** 2, dim=1).numpy()

            # LSTM Score
            lstm_loss = np.zeros(len(windows))
            if self.lstm and self.preprocessor.window_size > 1:
                X_lstm_in = torch.LongTensor(windows[:, :-1])
                y_target = torch.LongTensor(windows[:, -1])
                with torch.no_grad():
                    logits = self.lstm(X_lstm_in)
                    # Calculate Cross Entropy per sample
                    # We can use CrossEntropyLoss with reduction='none'
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    lstm_loss = criterion(logits, y_target).numpy()

            # Final Score
            total_score = ae_loss + lstm_loss
            
            # Aggregate per user (max anomaly or average)
            # We want specific abnormal sequences
            for i, score in enumerate(total_score):
                is_anomaly = score > 1.5 # Threshold (arbitrary for demo, normally tuned)
                if is_anomaly:
                    # Explainability: Extract Rule-Based Signals
                    log_window = user_logs.iloc[i : i + self.preprocessor.window_size]
                    explanations = self._extract_behavior_rules(log_window)
                    
                    details_str = "Abnormal Sequence Detected"
                    if explanations:
                        details_str += ": " + ", ".join(explanations)

                    results.append({
                        'User ID': user,
                        'Sequence_Start_Time': user_logs.iloc[i]['Timestamp'],
                        'Anomaly_Score': float(score),
                        'AE_Error': float(ae_loss[i]),
                        'LSTM_Deviation': float(lstm_loss[i]),
                        'Details': details_str
                    })
                    
        return pd.DataFrame(results)

    def _extract_behavior_rules(self, log_window):
        """
        Extracts rule-based indicators for Explainable AI (Time, Action, Sequence).
        """
        explanations = []
        timestamps = log_window['Timestamp'].dt.hour.values
        actions = log_window['Action'].str.upper().values
        resources = log_window['Resource'].str.lower().values
        
        # 1. Time-Based Signals
        # Midnight Access (00:00 - 05:00)
        if any((t >= 0) and (t < 5) for t in timestamps):
            explanations.append("Midnight/Early Morning Access")
        # Outside Business Hours (Assuming 9-18 is normal)
        if any((t < 9) or (t > 18) for t in timestamps):
            if "Midnight/Early Morning Access" not in explanations:
                explanations.append("Activity Outside Business Hours")

        # 2. Action Indicators
        if "DELETE" in actions:
            explanations.append("Deletion of Records")
        if "DOWNLOAD" in actions:
            # Check for sensitive resources
            if any(r in "".join(resources) for r in ["salary", "audit", "confidential", "secret", "password"]):
                explanations.append("Download of Sensitive/Confidential Files")
            else:
                explanations.append("Bulk Data Download")
        if "ACCESS" in actions and any("admin" in r for r in resources):
            explanations.append("Admin Panel Access")

        # 3. Sequence Patterns
        # Login -> Download -> Delete
        actions_str = " -> ".join(actions)
        if "LOGIN" in actions and "DOWNLOAD" in actions and "DELETE" in actions:
             explanations.append("Suspicious Sequence (Login->Download->Delete)")
        elif "DOWNLOAD" in actions and "DELETE" in actions:
             explanations.append("Data Exfiltration Pattern (Download->Delete)")
             
        # Privilege Escalation (Accessing admin without prior checks - simplified)
        if "ACCESS" in actions and any("admin" in r for r in resources) and "LOGIN" not in actions:
             # Just a heuristic for the window
             explanations.append("Potential Privilege Escalation")

        return explanations

if __name__ == "__main__":
    from ingestion import EvidenceIngestion
    ingestion = EvidenceIngestion()
    logs = ingestion.load_logs("data/sample_logs.csv")
    
    analyzer = BehaviorAnalyzer()
    # Force train for demo
    analyzer.train(logs)
    
    results = analyzer.analyze_behavior(logs)
    print(results)
