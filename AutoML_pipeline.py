# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:22:00 2026

@author: press
"""


import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  
import numpy as np                                                                   
import os

# ── Configuration ─────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(script_dir, "..", "data", "training_data.csv")
OUTPUT_DIR = os.path.join(script_dir, "..", "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "saved_model")

NUMERIC_COLS = ["U/Pb", "Nb", "Th/U", "La", "Rb", "Ba"]
TARGET = "Reservoir"

HOLDOUT_LOCATIONS = [
    {"location": "PrattWelker",  "reservoir": "DM"},
    {"location": "IndianMOR",    "reservoir": "DM"},
    {"location": "Macquarie",    "reservoir": "DM"},
    {"location": "Shatsky",      "reservoir": "DM"},
    {"location": "SWPacific",    "reservoir": "EM1"},
    {"location": "Idaho",        "reservoir": "EM1"},
    {"location": "Indian2",      "reservoir": "EM1"},
    {"location": "EastAsia",     "reservoir": "EM1"},
    {"location": "WestPac",      "reservoir": "EM1"},
    {"location": "30W",          "reservoir": "EM2"},
    {"location": "RedSea",       "reservoir": "EM2"},
    {"location": "Marquesas",    "reservoir": "EM2", "external_file": os.path.join(script_dir, "..", "data", "EM2_test_Marquesas.csv"),},
    {"location": "64E",          "reservoir": "HIMU"},
    {"location": "Ant",          "reservoir": "HIMU"},
    {"location": "100W",         "reservoir": "HIMU"},
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Helper: prepare a pandas DataFrame into an H2O frame ──────────────────────
def to_h2o(df):
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = df[TARGET].astype(str)
    frame = h2o.H2OFrame(df)
    frame[TARGET] = frame[TARGET].asfactor()
    for col in NUMERIC_COLS:
        frame[col] = frame[col].asnumeric()
    return frame

# ── Helper: print an H2O-style confusion matrix ───────────────────────────── NEW
def print_confusion_matrix(y_true, y_pred):
    labels        = sorted(set(y_true.astype(str).unique()))
    cm            = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df         = pd.DataFrame(
        cm,
        index   = [f"Actual: {l}"    for l in labels],
        columns = [f"Predicted: {l}" for l in labels]
    )
    row_totals    = cm.sum(axis=1)
    row_errors    = (row_totals - np.diag(cm)) / row_totals
    overall_row   = cm.sum(axis=0) - np.diag(cm)
    total_samples = cm.sum()
    total_errors  = (row_totals - np.diag(cm)).sum()
    overall_error = total_errors / total_samples

    col_w = max(len(f"Predicted: {l}") for l in labels) + 2
    row_w = max(len(f"Actual: {l}")    for l in labels) + 2

    header = (f"{'':>{row_w}}"
              + "".join(f"{f'Predicted: {l}':>{col_w}}" for l in labels)
              + f"{'Error':>{col_w}}")
    print(header)
    print("-" * len(header))
    for i, (idx, row) in enumerate(cm_df.iterrows()):
        err_str = f"{int(row_totals[i] - np.diag(cm)[i])}/{row_totals[i]} = {row_errors[i]:.3f}"
        print(f"{idx:<{row_w}}" + "".join(f"{v:>{col_w}}" for v in row) + f"{err_str:>{col_w}}")
    print("-" * len(header))
    overall_str = f"{int(total_errors)}/{total_samples} = {overall_error:.3f}"
    print(f"{'Overall':<{row_w}}"
          + "".join(f"{v:>{col_w}}" for v in overall_row)
          + f"{overall_str:>{col_w}}")
    print()

# ── Helper: run AutoML and return the leader ──────────────────────────────────
def run_automl(train_frame):
    aml = H2OAutoML(
        max_models                            = 50,
        seed                                  = 42,
        balance_classes                       = True,
        nfolds                                = 5,
        keep_cross_validation_predictions     = False,
        keep_cross_validation_models          = False,
        keep_cross_validation_fold_assignment = False,
    )
    aml.train(x=NUMERIC_COLS, y=TARGET, training_frame=train_frame)

    # Store leaderboard IDs now, before any removal
    leaderboard_ids = aml.leaderboard["model_id"].as_data_frame()["model_id"].tolist()
    leader          = aml.leader
    leader_id       = leader.model_id
    is_ensemble     = "StackedEnsemble" in leader_id

    if is_ensemble:
        # Get the base model IDs from the ensemble so we can protect them
        base_model_ids = set(m["name"] for m in leader.params["base_models"]["actual"])
    else:
        base_model_ids = set()

    # Remove non-leader models that are NOT base models of the leader
    for model_id in leaderboard_ids:
        if model_id != leader_id and model_id not in base_model_ids:
            try:
                h2o.remove(model_id)
            except Exception:
                pass  # already removed or not found — safe to ignore

    return leader

# ── 1 & 2. Train AutoML on full dataset and find best model ───────────────────
h2o.init()

raw = pd.read_csv(DATA_PATH)
h2o_full = to_h2o(raw.copy())
train_full, test_full = h2o_full.split_frame(ratios=[0.9], seed=1)

print("Training AutoML on full dataset...")
best_model = run_automl(train_full)
print(f"Best model: {best_model.model_id}")

# ── Save the best model to disk ───────────────────────────────────────────────
model_path = h2o.save_model(model=best_model, path=MODEL_DIR, force=True)
print(f"Model saved to: {model_path}")
with open(os.path.join(MODEL_DIR, "best_model_info.txt"), "w") as f:
    f.write(f"Model ID:   {best_model.model_id}\n")
    f.write(f"Model path: {model_path}\n")
print(f"Model info written to: {os.path.join(MODEL_DIR, 'best_model_info.txt')}")

# ── 3. Evaluate on internal test set ─────────────────────────────────────────
print("\n── Internal Test Set Performance ──")
test_df  = test_full.as_data_frame()
preds    = best_model.predict(test_full).as_data_frame()
acc_test = accuracy_score(test_df[TARGET], preds["predict"])
print(f"Accuracy: {acc_test:.4f}")
print(classification_report(test_df[TARGET], preds["predict"]))

# ── Confusion matrix ─────────────────────────────────────────────────────── NEW
print("\n── Confusion Matrix (H2O-style) ──")
print_confusion_matrix(test_df[TARGET], preds["predict"])

# ── 4 & 5. Evaluate on each holdout location ─────────────────────────────────
results          = []
all_true         = []
all_pred         = []
all_test_results = []

for case in HOLDOUT_LOCATIONS:
    loc, res = case["location"], case["reservoir"]
    ext_file = case.get("external_file")
    print(f"\n── Holdout: {res} | {loc} ──")

    mask       = (raw["Location"] == loc) & (raw["Reservoir"] == res)
    train_data = raw[~mask].copy()

    if ext_file:
        test_data = pd.read_csv(ext_file)
        print(f"  Using external test file: {ext_file}")
    else:
        test_data = raw[mask].copy()

    print(f"  Train n={len(train_data)}  Test n={len(test_data)}")

    h2o_train = to_h2o(train_data)
    h2o_test  = to_h2o(test_data)

    model = run_automl(h2o_train)
    td    = h2o_test.as_data_frame()
    pr    = model.predict(h2o_test).as_data_frame()   # predictions collected first
    acc   = accuracy_score(td[TARGET], pr["predict"])


    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(td[TARGET], pr["predict"]))

    results.append({"reservoir": res, "location": loc, "accuracy": acc,
                    "n_test": len(test_data)})
    all_true.extend(td[TARGET].tolist())
    all_pred.extend(pr["predict"].tolist())

    loc_rows = test_data[NUMERIC_COLS].copy().reset_index(drop=True)
    loc_rows.insert(0, "Location",            loc)
    loc_rows.insert(1, "Reservoir",           td[TARGET].values)
    loc_rows.insert(2, "Predicted_Reservoir", pr["predict"].values)
    all_test_results.append(loc_rows)
    
    # Safe to remove everything now that predictions are in hand
    try:
        h2o.remove(model)          # removes ensemble + its base models together
    except Exception:
        pass
    h2o.remove(h2o_train)
    h2o.remove(h2o_test)
    
# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── Summary ──")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

for res_type in summary_df["reservoir"].unique():
    sub = summary_df[summary_df["reservoir"] == res_type]["accuracy"]
    print(f"{res_type}: mean={sub.mean():.3f}  min={sub.min():.3f}  max={sub.max():.3f}")

# ── Pooled classification report ──────────────────────────────────────────────
print("\n── Pooled Holdout Classification Report (all locations combined) ──")
pooled_acc = accuracy_score(all_true, all_pred)
print(f"Pooled Accuracy: {pooled_acc:.4f}")
print(classification_report(all_true, all_pred))

# ── Pooled confusion matrix ───────────────────────────────────────────────── NEW
print("\n── Pooled Confusion Matrix (all holdout locations combined) ──")
print_confusion_matrix(pd.Series(all_true), pd.Series(all_pred))

# ══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════

# ── 6. all_predictions_combined.csv ──────────────────────────────────────────
if all_test_results:
    combined_results = pd.concat(all_test_results, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "all_predictions_combined.csv")
    combined_results.to_csv(combined_path, index=False)
    print(f"\nAll predictions combined and saved to: {combined_path}")

    combined_results["Reservoir"]           = combined_results["Reservoir"].astype(str)
    combined_results["Predicted_Reservoir"] = combined_results["Predicted_Reservoir"].astype(str)
    print("\nOverall Classification Report:")
    print(classification_report(combined_results["Reservoir"],
                                combined_results["Predicted_Reservoir"]))

# ── 7. summary_results.csv ───────────────────────────────────────────────────
summary_data = [{"Reservoir": r["reservoir"], "Location": r["location"],
                 "Accuracy":  r["accuracy"],  "N_Test":   r["n_test"]}
                for r in results]

if summary_data:
    summary_out  = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "summary_results.csv")
    summary_out.to_csv(summary_path, index=False)
    print(f"\nSummary results saved to: {summary_path}")
else:
    print("No summary data available to save.")

# ── 8. reservoir_summary.csv ─────────────────────────────────────────────────
summary_by_reservoir = {}
for res_type in ["DM", "EM1", "EM2", "HIMU"]:
    subset = [r for r in results if r["reservoir"] == res_type]
    if subset:
        accuracies = [r["accuracy"] for r in subset]
        summary_by_reservoir[res_type] = {
            "avg_accuracy":     sum(accuracies) / len(accuracies),
            "min_accuracy":     min(accuracies),
            "max_accuracy":     max(accuracies),
            "locations_tested": len(accuracies),
        }
    else:
        print(f"Warning: No results found for reservoir type {res_type}.")
        summary_by_reservoir[res_type] = {
            "avg_accuracy": None, "min_accuracy": None,
            "max_accuracy": None, "locations_tested": 0,
        }

print("\nSummary by Reservoir Type:")
for res_type, stats in summary_by_reservoir.items():
    if stats["avg_accuracy"] is not None:
        print(f"  {res_type}: Avg={stats['avg_accuracy']:.3f}  "
              f"Min={stats['min_accuracy']:.3f}  Max={stats['max_accuracy']:.3f}  "
              f"n_locations={stats['locations_tested']}")
    else:
        print(f"  {res_type}: No accuracy data available.")

reservoir_summary_df = pd.DataFrame([
    {"Reservoir":        res_type,
     "Average_Accuracy": stats["avg_accuracy"],
     "Min_Accuracy":     stats["min_accuracy"],
     "Max_Accuracy":     stats["max_accuracy"],
     "Locations_Tested": stats["locations_tested"]}
    for res_type, stats in summary_by_reservoir.items()
])
reservoir_summary_path = os.path.join(OUTPUT_DIR, "reservoir_summary.csv")
reservoir_summary_df.to_csv(reservoir_summary_path, index=False)
print(f"\nReservoir summary saved to: {reservoir_summary_path}")

print(f"\nAll outputs written to: {OUTPUT_DIR}")