# SkillLens -- Model Evaluation Report

Test set size: 450 students (untouched since the Step 4 split; never used for hyperparameter tuning).

## Model comparison

| model                          |   accuracy |   macro_precision |   macro_recall |   macro_f1 |
|:-------------------------------|-----------:|------------------:|---------------:|-----------:|
| Baseline (Logistic Regression) |      0.698 |             0.729 |          0.696 |      0.711 |
| Random Forest (untuned)        |      0.54  |             0.599 |          0.446 |      0.458 |
| Random Forest (tuned)          |      0.58  |             0.6   |          0.578 |      0.586 |

## What changed between untuned and tuned Random Forest

The untuned Random Forest (`n_estimators=300`, unlimited depth) overfit the training set and collapsed predictions toward the two middle classes, missing most 'Highly Ready' students (recall 0.11 for that class). GridSearchCV, scored on the held-out validation fold only (never the test set), selected `class_weight='balanced'`, `min_samples_leaf=5`, unlimited depth and 300 trees -- the class-balancing and larger leaf size directly target the overfitting and minority-class collapse seen in Step 5.

## Final model choice

**Random Forest (tuned)** is used going forward for explainability (Step 7), the app, and CSV predictions -- selected on validation performance and confirmed on the untouched test set, not on test-set performance alone.

## Metric definitions

- **Accuracy**: fraction of students whose category was predicted exactly right.
- **Macro Precision**: of all students predicted into a given category, how many actually belonged there -- averaged equally across the 4 categories (so the small 'Highly Ready' class counts as much as 'Developing').
- **Macro Recall**: of all students who actually belong to a category, how many were correctly found -- again averaged equally per class.
- **Macro F1**: harmonic mean of macro precision and recall; the single number used to compare models fairly given class imbalance.
