from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

def new_func():
    XGBOOST_AVAILABLE = False
    return XGBOOST_AVAILABLE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = new_func()


class MLEngine:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(max_iter=500, random_state=42)
        self.xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss') if XGBOOST_AVAILABLE else None
        self.gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Performance tracking
        self.model_scores = {}
        self.feature_importances = {}
        self.cluster_labels = None
        self.best_model_name = "Random Forest"
        self.best_model = None

    def train_credit_risk_models(self, df, target_col):
        try:
            X_raw = df.select_dtypes(include=[np.number]).copy()
            if X_raw.empty:
                return False, "No numeric data to train on."

            # Drop engineered columns from features to avoid leakage
            drop_cols = [c for c in ['Health_Score', 'Risk_Level'] if c in X_raw.columns]
            X_raw.drop(columns=drop_cols, inplace=True, errors='ignore')

            # Create classification target: 0=Low, 1=Medium, 2=High risk
            threshold_high = df[target_col].quantile(0.33)
            threshold_med  = df[target_col].quantile(0.66)
            y = df[target_col].apply(
                lambda val: 0 if val >= threshold_med else (1 if val >= threshold_high else 2)
            )

            X_scaled = self.scaler.fit_transform(X_raw)

            results = {}

            # --- Random Forest ---
            self.rf_model.fit(X_scaled, y)
            rf_preds = self.rf_model.predict(X_scaled)
            rf_acc = accuracy_score(y, rf_preds)
            rf_f1  = f1_score(y, rf_preds, average='weighted', zero_division=0)
            results['Random Forest'] = {'model': self.rf_model, 'accuracy': rf_acc, 'f1': rf_f1}

            # --- Logistic Regression ---
            self.lr_model.fit(X_scaled, y)
            lr_preds = self.lr_model.predict(X_scaled)
            lr_acc = accuracy_score(y, lr_preds)
            lr_f1  = f1_score(y, lr_preds, average='weighted', zero_division=0)
            results['Logistic Regression'] = {'model': self.lr_model, 'accuracy': lr_acc, 'f1': lr_f1}

            # --- XGBoost (if available) ---
            if XGBOOST_AVAILABLE:
                self.xgb_model.fit(X_scaled, y)
                xgb_preds = self.xgb_model.predict(X_scaled)
                xgb_acc = accuracy_score(y, xgb_preds)
                xgb_f1  = f1_score(y, xgb_preds, average='weighted', zero_division=0)
                results['XGBoost'] = {'model': self.xgb_model, 'accuracy': xgb_acc, 'f1': xgb_f1}

            # --- Gradient Boosting Regressor for Health Score ---
            self.gb_regressor.fit(X_scaled, df[target_col].values)

            # Store all model scores
            self.model_scores = {name: {'accuracy': v['accuracy'], 'f1': v['f1']} for name, v in results.items()}

            # Pick best model by accuracy
            best_name = max(results, key=lambda k: results[k]['accuracy'])
            self.best_model_name = best_name
            self.best_model = results[best_name]['model']

            # --- Feature Importances from RF ---
            feat_names = X_raw.columns.tolist()
            importances = self.rf_model.feature_importances_
            self.feature_importances = dict(sorted(
                zip(feat_names, importances), key=lambda x: x[1], reverse=True
            ))

            # --- KMeans Clustering for district segmentation ---
            self.cluster_labels = self.kmeans.fit_predict(X_scaled)

            self.is_trained = True
            msg = (
                f"Models Trained! Best: {best_name} "
                f"(Acc: {results[best_name]['accuracy']:.1%}, F1: {results[best_name]['f1']:.2f})"
            )
            return True, msg

        except Exception as e:
            return False, f"Engine Error: {e}"

    def predict_health_scores(self, df, target_col):
        """Use Gradient Boosting to predict health scores (0–100)."""
        X_raw = df.select_dtypes(include=[np.number]).copy()
        drop_cols = [c for c in ['Health_Score', 'Risk_Level'] if c in X_raw.columns]
        X_raw.drop(columns=drop_cols, inplace=True, errors='ignore')
        X_scaled = self.scaler.transform(X_raw)

        raw_scores = self.gb_regressor.predict(X_scaled)
        # Normalize to 0–100
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            scores = ((raw_scores - min_s) / (max_s - min_s) * 100).astype(int)
        else:
            scores = np.full(len(raw_scores), 50)
        return np.clip(scores, 0, 100)

    def get_district_segments(self):
        """Return cluster labels mapped to meaningful categories."""
        if self.cluster_labels is None:
            return None
        mapping = {0: "Emerging", 1: "Stable", 2: "Declining"}
        return [mapping[l] for l in self.cluster_labels]

    def get_model_report(self):
        """Return a formatted model performance report string."""
        if not self.model_scores:
            return "No models trained yet."
        lines = ["MODEL PERFORMANCE METRICS", "=" * 45]
        for name, scores in self.model_scores.items():
            marker = " ◀ BEST" if name == self.best_model_name else ""
            lines.append(f"  {name:<22} Acc: {scores['accuracy']:.1%}  F1: {scores['f1']:.2f}{marker}")
        lines.append("")
        lines.append("TOP FEATURE IMPORTANCES (Random Forest)")
        lines.append("-" * 45)
        for i, (feat, imp) in enumerate(list(self.feature_importances.items())[:8]):
            bar = "█" * int(imp * 40)
            lines.append(f"  {feat[:22]:<22} {imp:.3f} {bar}")
        return "\n".join(lines)