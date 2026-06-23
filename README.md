# Models

Lightweight implementations of **Linear Regression** and **Logistic Regression** from scratch using NumPy.

---

## Models

### LinearRegression
Gradient descent-based regression for continuous target prediction.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iters` | `10000` | Max training iterations |
| `alpha` | `0.01` | Learning rate |
| `tolerance` | `0.005` | Early stopping threshold |

**Methods:** `fit(X, y)` · `predict(X)` · `score(X, y)` → R² score

---

### LogisticRegression
Sigmoid + binary cross-entropy classifier for binary classification.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iters` | `10000` | Max training iterations |
| `alpha` | `0.01` | Learning rate |
| `tolerance` | `0.005` | Early stopping threshold |

**Methods:** `fit(X, y)` · `predict(X)` → class labels · `score(X, y)` → accuracy

---

## Usage

```python
from Models.models import LinearRegression, LogisticRegression
import numpy as np

# Linear Regression
reg = LinearRegression(n_iters=5000, alpha=0.01)
reg.fit(X_train, y_train, verbose=True)
print(reg.score(X_test, y_test))

# Logistic Regression
clf = LogisticRegression(n_iters=5000, alpha=0.01)
clf.fit(X_train, y_train, verbose=True)
print(clf.score(X_test, y_test))
```

## Requirements

```
numpy
```

---

> Both models support **early stopping** — training halts automatically when the cost change drops below `tolerance`.
