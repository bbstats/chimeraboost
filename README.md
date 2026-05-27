# chimeraboost
Full-featured gradient boosting library with a Python/numba backend, inspired by CatBoost

<img width="500" height="500" alt="ChatGPT Image May 26, 2026, 05_12_17 PM" src="https://github.com/user-attachments/assets/ee98a4e2-9fa7-4ef1-9e64-e398f398966c" />


* What?
    * GBDT library that only depends on numpy, numba, and scikit-learn
    * Near equivalent performance to CatBoost on CPU

* Why?
    * I want to be able to modify my GBDT library at will
    * I know Python and I don't know C

* How?


```
from chimeraboost import ChimeraBoostClassifier
clf = ChimeraBoostClassifier()
clf.fit(X, y)
```
