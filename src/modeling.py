"""
Predictive modeling for insurance claim severity and risk-based pricing
Includes Linear Regression, Random Forest, and XGBoost models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def _prepare_feature_matrix(model_df, feature_cols):
    X = model_df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely missing to avoid median-imputation producing NaNs.
    usable_cols = [col for col in X.columns if not X[col].isna().all()]
    X = X[usable_cols]

    if X.empty:
        raise ValueError("No usable numeric features remain after preprocessing")

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix still contains non-finite values after preprocessing")

    return X, usable_cols


def _sanitize_array(array_like):
    array = np.asarray(array_like, dtype=float)
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def prepare_modeling_data(df, target='TotalClaims', for_severity=True):
    """
    Prepare data for modeling
    
    Parameters:
    -----------
    df : DataFrame
        Insurance dataset
    target : str
        Target variable
    for_severity : bool
        If True, filter for claims > 0 (severity model)
        If False, use all data (probability model)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Filter data if needed
    if for_severity:
        model_df = df[df[target] > 0].copy()
        print(f"Severity model: {len(model_df)} policies with claims")
    else:
        model_df = df.copy()
        print(f"Probability model: {len(model_df)} total policies")
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    # Select features (exclude target and identifiers)
    exclude_cols = [target, 'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 
                    'LossRatio', 'Margin', 'HasClaim']
    feature_cols = [
        col for col in model_df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]
    
    # Handle missing and non-finite values
    X, usable_cols = _prepare_feature_matrix(model_df, feature_cols)
    feature_cols = usable_cols
    y = model_df[target]

    if X.isnull().any().any():
        raise ValueError("Severity feature matrix still contains missing values after preprocessing")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, feature_cols


def prepare_probability_model(df):
    """
    Prepare data for claim probability classification
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Target: whether claim occurred
    df['ClaimOccurred'] = (df['TotalClaims'] > 0).astype(int)
    
    if 'TotalClaims' not in df.columns:
        raise ValueError("Target column 'TotalClaims' not found in dataframe")

    # Exclude columns
    exclude_cols = ['TotalClaims', 'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth',
                    'LossRatio', 'Margin', 'HasClaim', 'ClaimOccurred']
    
    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]
    
    # Handle missing and non-finite values
    X, usable_cols = _prepare_feature_matrix(df, feature_cols)
    feature_cols = usable_cols
    y = df['ClaimOccurred']

    if X.isnull().any().any():
        raise ValueError("Probability feature matrix still contains missing values after preprocessing")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_severity_models(X_train, y_train):
    """
    Train regression models for claim severity prediction
    
    Returns:
    --------
    dict of trained models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    trained_models = {}
    X_train = _sanitize_array(X_train)
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models


def train_probability_model(X_train, y_train):
    """
    Train classification model for claim probability
    
    Returns:
    --------
    Trained classifier
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    X_train = _sanitize_array(X_train)
    model.fit(X_train, y_train)
    return model


def evaluate_regression_models(models, X_test, y_test):
    """
    Evaluate regression models and return comparison table
    """
    results = []
    X_test = _sanitize_array(X_test)
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
        
        results.append({
            'Model': name,
            'RMSE': f"R{rmse:,.0f}",
            'MAE': f"R{mae:,.0f}",
            'R²': f"{r2:.4f}",
            'CV R² (mean)': f"{cv_scores.mean():.4f}",
            'CV R² (std)': f"{cv_scores.std():.4f}"
        })
    
    return pd.DataFrame(results)


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate classification model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    X_test = _sanitize_array(X_test)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    return pd.DataFrame([results])


def calculate_risk_premium(claim_probability, predicted_severity, expense_loading=0.15, profit_margin=0.10):
    """
    Calculate risk-based premium
    
    Premium = (P(claim) × Predicted Severity) × (1 + Expense Loading + Profit Margin)
    
    Parameters:
    -----------
    claim_probability : float
        Probability of claim occurring (0 to 1)
    predicted_severity : float
        Predicted claim amount if claim occurs
    expense_loading : float
        Operating expenses as percentage (default 15%)
    profit_margin : float
        Desired profit margin (default 10%)
    
    Returns:
    --------
    float: Recommended premium
    """
    risk_premium = claim_probability * predicted_severity
    total_premium = risk_premium * (1 + expense_loading + profit_margin)
    
    return total_premium


def get_feature_importance_shap(model, X_train, feature_names, model_type='regression'):
    """
    Calculate SHAP values for model interpretability
    
    Returns:
    --------
    DataFrame with feature importance
    """
    import shap

    X_train = _sanitize_array(X_train)
    
    if model_type == 'regression':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:100])  # Use subset for speed
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:100])
    
    # Calculate mean absolute SHAP values
    if model_type == 'classification':
        shap_values = shap_values[:, :, 1] if len(shap_values.shape) > 2 else shap_values
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    return importance_df.head(10)