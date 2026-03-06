import os
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """Load and preprocess the insurance claims dataset."""
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Replace placeholder missing values
    df.replace('?', np.nan, inplace=True)
    logger.info("Dataset shape: %s", df.shape)

    # Encode target variable
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
    logger.info("Target variable encoded. Fraud cases: %d, Genuine cases: %d",
                df['fraud_reported'].sum(), len(df) - df['fraud_reported'].sum())

    # Drop columns that are not useful for modeling
    columns_to_drop = [
        'policy_number',
        'policy_bind_date',
        'insured_zip',
        'incident_location',
        'incident_date'
    ]
    df = df.drop(columns=columns_to_drop)
    logger.info("Dropped %d columns: %s", len(columns_to_drop), columns_to_drop)

    # One-hot encode categorical variables (drop_first to avoid collinearity)
    df_encoded = pd.get_dummies(df, drop_first=True)
    logger.info("After one-hot encoding: %d features", df_encoded.shape[1] - 1)

    X = df_encoded.drop('fraud_reported', axis=1)
    y = df_encoded['fraud_reported']

    # Impute missing numeric values with the median
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    logger.info("Missing values imputed with median strategy")

    return df_encoded, X.values, y.values, list(df_encoded.drop('fraud_reported', axis=1).columns)

def train_and_evaluate_model(X: np.ndarray, y: np.ndarray) -> Tuple[SVC, StandardScaler, Dict[str, Any]]:
    """Train SVM model and return model, scaler, and evaluation metrics."""
    logger.info("Splitting data into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Features scaled using StandardScaler")

    # Train SVM with balanced class weights
    model = SVC(class_weight='balanced', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    logger.info("SVM model trained with balanced class weights")

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'cross_val_scores': cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1').tolist()
    }

    logger.info("Model evaluation completed:")
    logger.info("ROC-AUC: %.3f", metrics['roc_auc'])
    logger.info("F1-Score: %.3f", metrics['f1_score'])
    logger.info("Cross-validation F1 scores: %s", metrics['cross_val_scores'])

    return model, scaler, metrics

def save_artifacts(model: SVC, scaler: StandardScaler, columns: list, metrics: Dict[str, Any]) -> None:
    """Save model artifacts and evaluation metrics."""
    artifacts = [
        ('svm_model.pkl', model),
        ('scaler.pkl', scaler),
        ('model_columns.pkl', columns),
        ('model_metrics.pkl', metrics)
    ]

    for filename, artifact in artifacts:
        with open(filename, 'wb') as f:
            pickle.dump(artifact, f)
        logger.info("Saved %s", filename)

def main():
    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "insurance_claims.csv")

    # Preprocess data
    df_encoded, X, y, columns = load_and_preprocess_data(csv_path)

    # Train and evaluate model
    model, scaler, metrics = train_and_evaluate_model(X, y)

    # Save artifacts
    save_artifacts(model, scaler, columns, metrics)

    logger.info("Training complete. All artifacts saved.")

if __name__ == '__main__':
    main()
