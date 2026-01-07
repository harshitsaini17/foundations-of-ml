# ML From Scratch (Paper → NumPy → Reality)

This repository is a disciplined, paper-first implementation of core machine learning algorithms using **only NumPy**.

The goal is not speed, benchmarks, or Kaggle scores.
The goal is **understanding**.

Every algorithm is:
- Derived from the original paper
- Implemented from scratch (NumPy only)
- Verified against scikit-learn
- Visualized and stress-tested
- Documented with failure modes

---

## Why This Exists

Most ML code today is written *on top* of abstractions.
This project strips those away.

If I can’t derive it, I don’t code it.
If I can’t explain it, I don’t ship it.

---

## Tech Stack

- Python
- NumPy
- Matplotlib
- scikit-learn (**verification only**)

No PyTorch. No TensorFlow. No shortcuts.

---

## Curriculum Progress

| Week | Topic | Status |
|----|------|------|
| 01 | Logistic Regression | 🟨 |
| 02 | Ridge Regression | ⬜ |
| 03 | Lasso Regression | ⬜ |
| 04 | Linear Discriminant Analysis | ⬜ |
| 05 | Support Vector Machines (Primal) | ⬜ |
| 06 | Principal Component Analysis | ⬜ |
| 07 | K-Means | ⬜ |
| 08 | Gaussian Mixture Models | ⬜ |
| 09 | AdaBoost | ⬜ |
| 10 | Gradient Boosting | ⬜ |

(⬜ → 🟨 → ✅ as you progress)

---

## Rules I Follow

- No copying implementations
- No training via sklearn
- All gradients derived manually
- All results verified independently
- Bugs documented, not hidden

---

## License
MIT
