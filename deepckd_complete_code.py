"""
DeepCKD-Net: Complete Implementation for Chronic Kidney Disease Prediction
Author: Research Team
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================================
# PART 1: DATA PREPROCESSING MODULE
# ============================================================================

class DataPreprocessor:
    """
    Module (a): Input preprocessing and feature engineering pipeline
    """
    def __init__(self, imputation_method='mice', scaling_method='zscore'):
        self.imputation_method = imputation_method
        self.scaling_method = scaling_method
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = None
        
    def fit(self, X, y=None):
        """Fit preprocessing parameters"""
        # Initialize imputer based on method
        if self.imputation_method == 'mice':
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
        else:
            self.imputer = KNNImputer(n_neighbors=5)
        
        # Separate numerical and categorical columns
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna('missing')
            le.fit(X[col])
            self.label_encoders[col] = le
            
        # Fit imputer and scaler on numerical features
        X_encoded = self._encode_categorical(X)
        X_imputed = self.imputer.fit_transform(X_encoded)
        self.scaler.fit(X_imputed)
        
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        X = X.copy()
        
        # Encode categorical variables
        X_encoded = self._encode_categorical(X)
        
        # Impute missing values
        X_imputed = self.imputer.transform(X_encoded)
        
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def _encode_categorical(self, X):
        """Encode categorical variables"""
        X = X.copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna('missing')
                X[col] = self.label_encoders[col].transform(X[col])
        return X[self.numerical_cols + self.categorical_cols]
    
    def engineer_features(self, X):
        """Create additional engineered features"""
        X = pd.DataFrame(X)
        
        # Create interaction features for key biomarkers
        if X.shape[1] >= 10:
            X['creatinine_urea_ratio'] = X.iloc[:, 0] / (X.iloc[:, 1] + 1e-8)
            X['hemoglobin_albumin_prod'] = X.iloc[:, 2] * X.iloc[:, 3]
            X['egfr_estimate'] = 175 * (X.iloc[:, 0] ** -1.154) * (X.iloc[:, 5] ** -0.203)
            
        return X.values

# ============================================================================
# PART 2: HIERARCHICAL TRANSFORMER ENCODER MODULE
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.W_o(context)
        return output

class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class HierarchicalTransformerEncoder(nn.Module):
    """
    Module (b): Hierarchical transformer encoder with multi-head attention
    """
    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.3):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Embed input
        x = self.input_embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        h_trans = self.output_projection(x)
        return h_trans

# ============================================================================
# PART 3: GRADIENT BOOSTING ENSEMBLE MODULE
# ============================================================================

class GradientBoostingEnsemble(nn.Module):
    """
    Module (c): Gradient boosting ensemble module
    """
    def __init__(self, input_dim, n_estimators=100, learning_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        # Neural network representation of gradient boosting
        self.weak_learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            ) for _ in range(min(10, n_estimators))  # Use 10 learners for efficiency
        ])
        
        self.aggregator = nn.Linear(16 * min(10, n_estimators), 256)
        
    def forward(self, x):
        # Get predictions from weak learners
        predictions = []
        for learner in self.weak_learners:
            pred = learner(x)
            predictions.append(pred)
        
        # Concatenate all predictions
        combined = torch.cat(predictions, dim=-1)
        
        # Aggregate with learned weights
        h_boost = self.aggregator(combined)
        return h_boost

# ============================================================================
# PART 4: ADAPTIVE FUSION MECHANISM
# ============================================================================

class AdaptiveFusionModule(nn.Module):
    """
    Module (d): Adaptive fusion mechanism
    """
    def __init__(self, trans_dim=256, boost_dim=256, fusion_dim=512):
        super().__init__()
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(trans_dim + boost_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.trans_projection = nn.Linear(trans_dim, fusion_dim)
        self.boost_projection = nn.Linear(boost_dim, fusion_dim)
        
    def forward(self, h_trans, h_boost):
        # Project to same dimension
        h_trans_proj = self.trans_projection(h_trans)
        h_boost_proj = self.boost_projection(h_boost)
        
        # Compute gating weights
        combined = torch.cat([h_trans, h_boost], dim=-1)
        gate_weights = self.gate(combined)
        
        # Weighted fusion
        h_fused = gate_weights * h_trans_proj + (1 - gate_weights) * h_boost_proj
        return h_fused

# ============================================================================
# PART 5: CONFIDENCE-AWARE PREDICTION LAYER
# ============================================================================

class ConfidenceAwarePrediction(nn.Module):
    """
    Module (e): Confidence-aware prediction layer
    """
    def __init__(self, input_dim=512, n_classes=2, mc_dropout_rate=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(mc_dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(mc_dropout_rate),
            nn.Linear(128, n_classes)
        )
        
        self.mc_dropout_rate = mc_dropout_rate
        self.n_mc_samples = 50
        
    def forward(self, x, training=False):
        if training:
            # Standard forward pass during training
            logits = self.classifier(x)
            return logits
        else:
            # Monte Carlo dropout for uncertainty estimation
            predictions = []
            
            # Enable dropout during inference
            for module in self.classifier.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
            
            # Multiple forward passes
            for _ in range(self.n_mc_samples):
                with torch.no_grad():
                    pred = self.classifier(x)
                    predictions.append(torch.softmax(pred, dim=-1))
            
            # Stack predictions
            predictions = torch.stack(predictions)
            
            # Mean prediction
            mean_pred = predictions.mean(dim=0)
            
            # Uncertainty (entropy)
            entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=-1)
            max_entropy = np.log(mean_pred.shape[-1])
            confidence = 1 - (entropy / max_entropy)
            
            return mean_pred, confidence

# ============================================================================
# PART 6: COMPLETE DEEPCKD-NET MODEL
# ============================================================================

class DeepCKDNet(nn.Module):
    """
    Complete DeepCKD-Net architecture integrating all modules
    """
    def __init__(self, input_dim, n_classes=2):
        super().__init__()
        
        # Initialize all modules
        self.transformer = HierarchicalTransformerEncoder(input_dim)
        self.boosting = GradientBoostingEnsemble(input_dim)
        self.fusion = AdaptiveFusionModule()
        self.prediction = ConfidenceAwarePrediction(n_classes=n_classes)
        
    def forward(self, x, return_confidence=False):
        # Pass through transformer and boosting
        h_trans = self.transformer(x)
        h_boost = self.boosting(x)
        
        # Adaptive fusion
        h_fused = self.fusion(h_trans, h_boost)
        
        # Prediction
        if self.training:
            output = self.prediction(h_fused, training=True)
            return output
        else:
            if return_confidence:
                predictions, confidence = self.prediction(h_fused, training=False)
                return predictions, confidence
            else:
                output = self.prediction(h_fused, training=True)
                return output

# ============================================================================
# PART 7: TRAINING MODULE
# ============================================================================

class DeepCKDTrainer:
    """
    Training module for DeepCKD-Net
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader, epochs=100, early_stopping_patience=10):
        """Train the model"""
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_deepckd_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_deepckd_model.pth'))
        
        return self.history

# ============================================================================
# PART 8: EVALUATION MODULE
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive evaluation module with metrics and visualization
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        
    def predict_with_confidence(self, test_loader):
        """Get predictions with confidence scores"""
        self.model.eval()
        all_predictions = []
        all_confidences = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                predictions, confidence = self.model(data, return_confidence=True)
                
                all_predictions.append(predictions.cpu().numpy())
                all_confidences.append(confidence.cpu().numpy())
                all_targets.append(target.numpy())
        
        predictions = np.concatenate(all_predictions)
        confidences = np.concatenate(all_confidences)
        targets = np.concatenate(all_targets)
        
        return predictions, confidences, targets
    
    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """Compute comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-CKD', 'CKD'],
                   yticklabels=['Non-CKD', 'CKD'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

# ============================================================================
# PART 9: SHAP INTERPRETABILITY MODULE
# ============================================================================

class SHAPInterpreter:
    """
    Module (f): SHAP-based interpretability module
    """
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data
        
    def explain_predictions(self, X_test, feature_names=None):
        """Generate SHAP explanations"""
        # Create a wrapper function for SHAP
        def model_predict(x):
            x_tensor = torch.FloatTensor(x)
            self.model.eval()
            with torch.no_grad():
                output = self.model(x_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                probs = torch.softmax(output, dim=-1)
            return probs.numpy()
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(model_predict, self.background_data[:100])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test[:10])  # Explain first 10 samples
        
        # Visualize feature importance
        if feature_names:
            self.plot_feature_importance(shap_values, X_test[:10], feature_names)
        
        return shap_values
    
    def plot_feature_importance(self, shap_values, X_test, feature_names):
        """Plot SHAP feature importance"""
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()

# ============================================================================
# PART 10: MAIN PIPELINE
# ============================================================================

def load_ckd_data(filepath='kidney_disease.csv'):
    """Load and prepare CKD dataset"""
    # Load data
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop(['classification', 'id'], axis=1, errors='ignore')
    y = df['classification'].apply(lambda x: 1 if 'ckd' in str(x).lower() else 0)
    
    return X, y

def main():
    """
    Main execution pipeline for DeepCKD-Net
    """
    print("=" * 60)
    print("DeepCKD-Net: Advanced CKD Prediction System")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    X, y = load_ckd_data()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Engineer features
    X_train_processed = preprocessor.engineer_features(X_train_processed)
    X_val_processed = preprocessor.engineer_features(X_val_processed)
    X_test_processed = preprocessor.engineer_features(X_test_processed)
    
    print(f"  Training samples: {X_train_processed.shape[0]}")
    print(f"  Validation samples: {X_val_processed.shape[0]}")
    print(f"  Test samples: {X_test_processed.shape[0]}")
    print(f"  Feature dimension: {X_train_processed.shape[1]}")
    
    # 2. Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_processed),
        torch.LongTensor(y_train.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_processed),
        torch.LongTensor(y_val.values)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_processed),
        torch.LongTensor(y_test.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Initialize model
    print("\n[2] Initializing DeepCKD-Net model...")
    model = DeepCKDNet(input_dim=X_train_processed.shape[1], n_classes=2)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Train model
    print("\n[3] Training model...")
    trainer = DeepCKDTrainer(model)
    history = trainer.fit(train_loader, val_loader, epochs=50, early_stopping_patience=10)
    
    # 5. Evaluate model
    print("\n[4] Evaluating model...")
    evaluator = ModelEvaluator(model)
    
    # Get predictions with confidence
    predictions, confidences, targets = evaluator.predict_with_confidence(test_loader)
    y_pred = np.argmax(predictions, axis=1)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(targets, y_pred, predictions)
    
    print("\n[5] Performance Metrics:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Average confidence
    avg_confidence = np.mean(confidences)
    print(f"  Average Confidence: {avg_confidence:.4f}")
    
    # 6. Plot results
    print("\n[6] Generating visualizations...")
    
    # Confusion Matrix
    evaluator.plot_confusion_matrix(targets, y_pred)
    
    # ROC Curve
    if 'auc_roc' in metrics:
        evaluator.plot_roc_curve(targets, predictions)
    
    # Training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 7. SHAP Interpretability
    print("\n[7] Generating SHAP explanations...")
    interpreter = SHAPInterpreter(model, X_train_processed[:100])
    feature_names = [f'Feature_{i}' for i in range(X_train_processed.shape[1])]
    shap_values = interpreter.explain_predictions(X_test_processed, feature_names)
    
    print("\n" + "=" * 60)
    print("DeepCKD-Net training and evaluation complete!")
    print("=" * 60)
    
    return model, metrics, history

# ============================================================================
# PART 11: K-FOLD CROSS VALIDATION
# ============================================================================

def cross_validate_model(X, y, n_splits=10):
    """
    Perform k-fold cross-validation
    """
    print(f"\n[K-Fold Cross Validation with {n_splits} folds]")
    print("=" * 50)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\nFold {fold}/{n_splits}:")
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Preprocess
        preprocessor = DataPreprocessor()
        preprocessor.fit(X_train_fold)
        
        X_train_processed = preprocessor.transform(X_train_fold)
        X_val_processed = preprocessor.transform(X_val_fold)
        
        X_train_processed = preprocessor.engineer_features(X_train_processed)
        X_val_processed = preprocessor.engineer_features(X_val_processed)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_processed),
            torch.LongTensor(y_train_fold.values)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_processed),
            torch.LongTensor(y_val_fold.values)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train model
        model = DeepCKDNet(input_dim=X_train_processed.shape[1], n_classes=2)
        trainer = DeepCKDTrainer(model)
        trainer.fit(train_loader, val_loader, epochs=30, early_stopping_patience=5)
        
        # Evaluate
        val_loss, val_acc = trainer.validate(val_loader)
        cv_scores.append(val_acc)
        print(f"  Fold {fold} Accuracy: {val_acc:.2f}%")
    
    # Print CV results
    print("\n" + "=" * 50)
    print("Cross-Validation Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.2f}%")
    print(f"  Std Deviation: {np.std(cv_scores):.2f}%")
    print(f"  Min Accuracy: {np.min(cv_scores):.2f}%")
    print(f"  Max Accuracy: {np.max(cv_scores):.2f}%")
    
    return cv_scores

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run main pipeline
    model, metrics, history = main()
    
    # Optional: Run cross-validation
    # X, y = load_ckd_data()
    # cv_scores = cross_validate_model(X, y, n_splits=10)