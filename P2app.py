"""
CIS 412 — Team Project Phase 2 (Rebuilt)
Gym Member Calorie Prediction via Regression Models
CRISP-DM Framework | W. P. Carey School of Business | ASU

FIXES APPLIED vs. ORIGINAL SUBMISSION:
1. OneHotEncoder for nominal categoricals (Gender, Workout_Type) — not LabelEncoder.
   LabelEncoder implies a numeric ordering (Male=0 < Female=1) that has no meaning for
   linear models and artificially inflates or deflates coefficients.
2. Proper sklearn Pipeline + ColumnTransformer — preprocessing fitted ONLY on training
   data to prevent leakage.
3. Correct tree-model terminology — regression trees minimize squared error / reduce
   variance; they do NOT maximize classification "information gain" / entropy.
4. GridSearchCV actually executed on GradientBoostingRegressor.
5. Business claims limited to prediction and association — no causal language.
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works in Colab & CI)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, learning_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/home/claude/phase2_project"
FIG_DIR    = os.path.join(BASE_DIR, "figures")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_PATH  = os.path.join(BASE_DIR, "gym_members_exercise_tracking.csv")

st.title("Gym Calorie Prediction App")
st.write("Predict calories burned based on your activity data!")

for d in [FIG_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — BUSINESS UNDERSTANDING  (see report for narrative)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: predict Calories_Burned (continuous) from biometric + session features.
# Supervised regression | CRISP-DM framework | evaluation: R², MAE, RMSE, 5-CV.

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2 — DATA UNDERSTANDING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nTarget summary:\n{df['Calories_Burned'].describe()}")

# ── EDA Figures ───────────────────────────────────────────────────────────────
# FIG 1: Target distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["Calories_Burned"], bins=30, color="#c0392b", edgecolor="white", alpha=0.85)
ax.axvline(df["Calories_Burned"].mean(), color="gold", linewidth=2,
           label=f'Mean = {df["Calories_Burned"].mean():.0f}')
ax.set_xlabel("Calories Burned (kcal)"); ax.set_ylabel("Count")
ax.set_title("Distribution: Calories Burned"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_target_distribution.png"), dpi=150)
plt.close()

# FIG 2: Scatter plots for top expected predictors
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
scatter_feats = [("Session_Duration (hours)", "#8b0000"),
                 ("Avg_BPM",                  "#b8860b")]
for ax, (feat, col) in zip(axes, scatter_feats):
    ax.scatter(df[feat], df["Calories_Burned"], alpha=0.4, color=col, s=12)
    ax.set_xlabel(feat); ax.set_ylabel("Calories Burned")
    ax.set_title(f"{feat} vs Calories Burned")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_scatter_top_features.png"), dpi=150)
plt.close()

# FIG 3: Correlation heatmap (numeric only)
fig, ax = plt.subplots(figsize=(10, 8))
num_df = df.select_dtypes(include=np.number)
mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
sns.heatmap(num_df.corr(), mask=mask, cmap="RdYlGn", annot=False,
            linewidths=0.4, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_correlation_heatmap.png"), dpi=150)
plt.close()

print("\nEDA figures saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — DATA PREPARATION")
print("=" * 60)

TARGET = "Calories_Burned"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify feature groups
CATEGORICAL_FEATURES = ["Gender", "Workout_Type"]   # explicitly named — safer
NUMERIC_FEATURES = [c for c in X.columns if c not in CATEGORICAL_FEATURES]

print(f"\nNumeric features  : {NUMERIC_FEATURES}")
print(f"Categorical feats : {CATEGORICAL_FEATURES}")

# Train / Test split — 80/20, random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── PREPROCESSING PIPELINES ───────────────────────────────────────────────────
# WHY OneHotEncoder for Gender & Workout_Type (not LabelEncoder):
#   LabelEncoder assigns arbitrary integers (Cardio=0, HIIT=1, Strength=2, Yoga=3).
#   For linear models this implies Yoga > Strength > HIIT > Cardio numerically,
#   which is meaningless and biases coefficients. OneHotEncoder creates binary
#   dummy columns so the model treats each category independently.
#   Tree models (RF, GB) are split-based and order-invariant, but OHE is still
#   correct practice; it just doesn't matter for their performance.

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())  # required for linear models; harmless for trees
])

categorical_pipeline = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Linear/Ridge preprocessor — scales numeric AND one-hot encodes categoricals
linear_preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,      NUMERIC_FEATURES),
    ("cat", categorical_pipeline,  CATEGORICAL_FEATURES),
])

# Tree preprocessor — one-hot encodes categoricals; no scaling needed for trees
# (tree splits are rank-order based, so scaling has no effect on split quality)
tree_preprocessor = ColumnTransformer([
    ("num", "passthrough",         NUMERIC_FEATURES),
    ("cat", categorical_pipeline,  CATEGORICAL_FEATURES),
])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODELING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — MODELING")
print("=" * 60)

# ── Build full pipelines ──────────────────────────────────────────────────────
# CORRECT TERMINOLOGY NOTE:
#   LinearRegression   — minimizes sum of squared residuals (OLS)
#   Ridge              — minimizes OLS loss + λ·Σw²  (L2 penalty shrinks coefficients)
#   RandomForest       — bagging ensemble; each split minimizes MSE / variance reduction;
#                        predictions are averaged across trees (NOT classification entropy)
#   GradientBoosting   — boosting: each new tree fits the *residuals* of the previous
#                        ensemble stage by stage; optimizes squared error loss function.

lr_pipe  = Pipeline([("prep", linear_preprocessor), ("model", LinearRegression())])
rr_pipe  = Pipeline([("prep", linear_preprocessor), ("model", Ridge(alpha=1.0))])
rf_pipe  = Pipeline([("prep", tree_preprocessor),
                     ("model", RandomForestRegressor(
                         n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))])
gb_pipe  = Pipeline([("prep", tree_preprocessor),
                     ("model", GradientBoostingRegressor(
                         n_estimators=200, random_state=42))])

models = {
    "Linear Regression":    lr_pipe,
    "Ridge Regression":     rr_pipe,
    "Random Forest":        rf_pipe,
    "Gradient Boosting":    gb_pipe,
}

# ── Train all models ──────────────────────────────────────────────────────────
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    print(f"  Trained: {name}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — EVALUATION")
print("=" * 60)

def evaluate(pipe, X_tr, y_tr, X_te, y_te, cv=5):
    """Returns dict of all required metrics."""
    y_tr_pred = pipe.predict(X_tr)
    y_te_pred = pipe.predict(X_te)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
    return {
        "Train R2":   r2_score(y_tr, y_tr_pred),
        "Test R2":    r2_score(y_te, y_te_pred),
        "Train MAE":  mean_absolute_error(y_tr, y_tr_pred),
        "Test MAE":   mean_absolute_error(y_te, y_te_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_tr, y_tr_pred)),
        "Test RMSE":  np.sqrt(mean_squared_error(y_te, y_te_pred)),
        "CV R2 Mean": cv_scores.mean(),
        "CV R2 Std":  cv_scores.std(),
    }

results = {}
for name, pipe in models.items():
    results[name] = evaluate(pipe, X_train, y_train, X_test, y_test)

metrics_df = pd.DataFrame(results).T.round(4)
print(f"\n{metrics_df.to_string()}")

# Save metrics
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))

# ── Comparison figures ────────────────────────────────────────────────────────
model_names = list(results.keys())
short_names  = ["Lin Reg", "Ridge", "Rand Forest", "Grad Boost"]
x = np.arange(len(model_names))
w = 0.35

# FIG 4: R² Train vs Test
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, [results[m]["Train R2"] for m in model_names], w,
       label="Train", color="#8b0000")
ax.bar(x + w/2, [results[m]["Test R2"]  for m in model_names], w,
       label="Test",  color="#b8860b")
ax.set_xticks(x); ax.set_xticklabels(short_names)
ax.set_ylabel("R²"); ax.set_title("R² Score: Train vs Test")
ax.set_ylim(0.9, 1.01); ax.legend()
for i, m in enumerate(model_names):
    ax.text(i - w/2, results[m]["Train R2"] + 0.001,
            f'{results[m]["Train R2"]:.4f}', ha='center', fontsize=7)
    ax.text(i + w/2, results[m]["Test R2"]  + 0.001,
            f'{results[m]["Test R2"]:.4f}',  ha='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_r2_train_test.png"), dpi=150)
plt.close()

# FIG 5: MAE comparison
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, [results[m]["Train MAE"] for m in model_names], w,
       label="Train", color="#8b0000")
ax.bar(x + w/2, [results[m]["Test MAE"]  for m in model_names], w,
       label="Test",  color="#b8860b")
ax.set_xticks(x); ax.set_xticklabels(short_names)
ax.set_ylabel("MAE (kcal)"); ax.set_title("Mean Absolute Error: Train vs Test")
ax.legend()
for i, m in enumerate(model_names):
    ax.text(i - w/2, results[m]["Train MAE"] + 0.3,
            f'{results[m]["Train MAE"]:.1f}', ha='center', fontsize=8)
    ax.text(i + w/2, results[m]["Test MAE"]  + 0.3,
            f'{results[m]["Test MAE"]:.1f}',  ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_mae_comparison.png"), dpi=150)
plt.close()

# FIG 6: RMSE comparison
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, [results[m]["Train RMSE"] for m in model_names], w,
       label="Train", color="#8b0000")
ax.bar(x + w/2, [results[m]["Test RMSE"]  for m in model_names], w,
       label="Test",  color="#b8860b")
ax.set_xticks(x); ax.set_xticklabels(short_names)
ax.set_ylabel("RMSE (kcal)"); ax.set_title("RMSE: Train vs Test")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_rmse_comparison.png"), dpi=150)
plt.close()

# FIG 7: 5-fold CV R² with error bars
fig, ax = plt.subplots(figsize=(8, 5))
cv_means = [results[m]["CV R2 Mean"] for m in model_names]
cv_stds  = [results[m]["CV R2 Std"]  for m in model_names]
ax.bar(short_names, cv_means, color=["#8b0000","#8b0000","#c0392b","#b8860b"],
       edgecolor="white", width=0.55)
ax.errorbar(short_names, cv_means, yerr=cv_stds, fmt="none",
            color="black", capsize=6, linewidth=2)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, m + s + 0.001, f'{m:.4f}\n±{s:.4f}', ha='center', fontsize=8)
ax.set_ylabel("5-Fold CV R²"); ax.set_title("5-Fold Cross-Validation R² (Mean ± Std)")
ax.set_ylim(0.9, 1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_cv_comparison.png"), dpi=150)
plt.close()

print("Comparison figures saved.")

# ── Best model predicted vs actual + residuals ────────────────────────────────
best_pipe = models["Gradient Boosting"]
y_pred_test = best_pipe.predict(X_test)

# FIG 12: Predicted vs Actual
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_test, alpha=0.5, color="#c0392b", s=18, label="Predictions")
lims = [y_test.min(), y_test.max()]
ax.plot(lims, lims, "--", color="gold", linewidth=1.5, label="Perfect Fit")
ax.set_xlabel("Actual Calories Burned"); ax.set_ylabel("Predicted Calories Burned")
ax.set_title(f"Gradient Boosting — Predicted vs Actual\n"
             f"R²={results['Gradient Boosting']['Test R2']:.4f}  "
             f"MAE={results['Gradient Boosting']['Test MAE']:.2f} kcal")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "12_predicted_vs_actual_gb.png"), dpi=150)
plt.close()

# FIG 13: Residuals
residuals = y_test.values - y_pred_test
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(residuals, bins=30, color="#8b0000", edgecolor="white", alpha=0.85)
ax.axvline(0,               color="gold",  linewidth=2, linestyle="--", label="Zero Error")
ax.axvline(residuals.mean(), color="white", linewidth=1.5, linestyle=":",
           label=f"Mean={residuals.mean():.1f}")
ax.set_xlabel("Residuals (kcal)"); ax.set_ylabel("Count")
ax.set_title(f"Gradient Boosting Residual Distribution\n"
             f"Mean={residuals.mean():.2f}  Std={residuals.std():.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "13_residual_plot_gb.png"), dpi=150)
plt.close()

# Save test predictions
pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_test,
              "Residual": residuals}).to_csv(
    os.path.join(OUTPUT_DIR, "predictions_test.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GRIDSEARCHCV  (required by rubric)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — GRIDSEARCHCV")
print("=" * 60)

# WHY GridSearchCV on TRAINING DATA only:
#   Using the test set to tune hyperparameters would cause data leakage — the
#   model would effectively be tuned to the test set, making evaluation optimistic.
#   GridSearchCV uses k-fold CV on the training data only to estimate how each
#   hyperparameter combination generalizes.

gb_param_grid = {
    "model__n_estimators":  [100, 200],
    "model__learning_rate": [0.05, 0.10],
    "model__max_depth":     [3, 5],
    "model__subsample":     [0.8, 1.0],
}

gs_gb = GridSearchCV(
    gb_pipe,
    param_grid=gb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=0,
    refit=True,
)
gs_gb.fit(X_train, y_train)

print(f"\nBest GradientBoosting params : {gs_gb.best_params_}")
print(f"Best CV R²                   : {gs_gb.best_score_:.4f}")

# Evaluate tuned model on test set
y_pred_tuned = gs_gb.best_estimator_.predict(X_test)
print(f"Tuned model — Test R²  : {r2_score(y_test, y_pred_tuned):.4f}")
print(f"Tuned model — Test MAE : {mean_absolute_error(y_test, y_pred_tuned):.2f}")

# Save GridSearch results
gs_results_df = pd.DataFrame(gs_gb.cv_results_)
gs_results_df.to_csv(os.path.join(OUTPUT_DIR, "gridsearch_results.csv"), index=False)

# FIG 15: GridSearchCV summary — mean test score per param combo
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(gs_results_df)), gs_results_df["mean_test_score"],
       color=["#b8860b" if i == gs_gb.best_index_ else "#8b0000"
              for i in range(len(gs_results_df))])
ax.set_xlabel("Hyperparameter Combination Index")
ax.set_ylabel("5-CV Mean R²")
ax.set_title("GridSearchCV Results — GradientBoostingRegressor\n"
             f"Best: {gs_gb.best_params_}  →  R²={gs_gb.best_score_:.4f}")
ax.axhline(gs_gb.best_score_, color="gold", linestyle="--", linewidth=1.5,
           label="Best Score")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "15_gridsearch_results.png"), dpi=150)
plt.close()

print("GridSearchCV figure saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — LEARNING CURVES (all 4 models)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — LEARNING CURVES")
print("=" * 60)

lc_models = {
    "Linear Regression": lr_pipe,
    "Ridge Regression":  rr_pipe,
    "Random Forest":     rf_pipe,
    "Gradient Boosting": gb_pipe,
}
fig_indices = [8, 9, 10, 11]

for (name, pipe), fig_idx in zip(lc_models.items(), fig_indices):
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train,
        cv=5, scoring="r2", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, "o-", color="#8b0000", label="Train R²")
    ax.plot(train_sizes, val_mean,   "o-", color="#b8860b", label="Test R²")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#8b0000")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#b8860b")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R²")
    ax.set_title(f"{name} — Learning Curve")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fname = f"{fig_idx:02d}_learning_curve_{name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8 — FEATURE IMPORTANCE")
print("=" * 60)

# CORRECT TERMINOLOGY:
#   GradientBoostingRegressor.feature_importances_ = cumulative reduction in MSE
#   attributed to splits on each feature across all trees, normalized to sum=1.
#   This is NOT classification "information gain" / entropy reduction.

gb_model   = best_pipe.named_steps["model"]
prep_step  = best_pipe.named_steps["prep"]
ohe_names  = list(
    prep_step.named_transformers_["cat"]
    .named_steps["ohe"]
    .get_feature_names_out(CATEGORICAL_FEATURES)
)
feature_names = NUMERIC_FEATURES + ohe_names

importances = gb_model.feature_importances_
feat_imp_df = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
               .sort_values("Importance", ascending=False)
               .reset_index(drop=True))
print(feat_imp_df.head(10).to_string(index=False))

# CAVEAT: feature importance reflects association, NOT causation.
feat_imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

# FIG 14: Feature importance plot
fig, ax = plt.subplots(figsize=(10, 7))
top_n = feat_imp_df.head(14)
colors = ["#b8860b" if v > feat_imp_df["Importance"].mean() else "#8b0000"
          for v in top_n["Importance"]]
ax.barh(top_n["Feature"][::-1], top_n["Importance"][::-1], color=colors[::-1])
ax.axvline(feat_imp_df["Importance"].mean(), color="gold", linestyle="--",
           linewidth=1.5, label="Mean Importance")
ax.set_xlabel("Importance (MSE Reduction — NOT causation)")
ax.set_title("Gradient Boosting — Feature Importance\n"
             "(cumulative reduction in squared error per feature)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "14_feature_importance.png"), dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(best_pipe, os.path.join(MODEL_DIR, "final_model.joblib"))
print(f"\nModel saved → {os.path.join(MODEL_DIR, 'final_model.joblib')}")

# STREAMLIT USER INPUT FORM
st.header("Predict Calories Burned")

age = st.slider("Age", 18, 59, 25)
weight = st.slider("Weight (kg)", 40.0, 130.0, 70.0)
height = st.slider("Height (m)", 1.5, 2.0, 1.75)
avg_bpm = st.slider("Average BPM", 120, 169, 140)
session_duration = st.slider("Session Duration (hours)", 0.5, 2.0, 1.0)

if st.button("Predict Calories"):
    sample = pd.DataFrame({
        "Age": [age],
        "Gender": ["Male"],
        "Weight (kg)": [weight],
        "Height (m)": [height],
        "Max_BPM": [180],
        "Avg_BPM": [avg_bpm],
        "Resting_BPM": [60],
        "Session_Duration (hours)": [session_duration],
        "Workout_Type": ["Cardio"],
        "Fat_Percentage": [20.0],
        "Water_Intake (liters)": [2.5],
        "Workout_Frequency (days/week)": [3],
        "Experience_Level": [2],
        "BMI": [weight / (height**2)]
    })

    pred = best_pipe.predict(sample)[0]
    st.success(f"Estimated Calories Burned: {pred:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nBest Model : Gradient Boosting")
print(f"  Test R²  : {results['Gradient Boosting']['Test R2']:.4f}")
print(f"  Test MAE : {results['Gradient Boosting']['Test MAE']:.2f} kcal")
print(f"  Test RMSE: {results['Gradient Boosting']['Test RMSE']:.2f} kcal")
print(f"  5-CV R²  : {results['Gradient Boosting']['CV R2 Mean']:.4f} ± "
      f"{results['Gradient Boosting']['CV R2 Std']:.4f}")
print(f"\nGridSearch Best Params: {gs_gb.best_params_}")
print(f"GridSearch Best CV R² : {gs_gb.best_score_:.4f}")
print("\nAll figures, metrics, and model artifacts saved.")
print("=" * 60)
