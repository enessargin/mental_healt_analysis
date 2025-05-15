# Mental Health Analysis 

A small, end‑to‑end project that explores a mental‑health survey dataset, trains classification models to predict **mental‑health risk**, and saves a production‑ready model artifact.  
Two complementary entry‑points are provided:

| Entry‑point | Purpose | Typical user |
|-------------|---------|--------------|
| **`Mental_Health_Analysis.ipynb`** | Exploratory data analysis (EDA), interactive model prototyping, rich visualisations. | Data scientists, analysts, demo audiences. |
| **`mental_healt_analysis.py`** | Repeatable, head‑less training pipeline that cleans data, selects & evaluates models, and serialises the best model to disk. | MLOps/engineering pipelines, CI/CD, batch re‑training. |

---

## Repository layout

```
.
├── Mental_Health_Analysis.ipynb   # Notebook for EDA & experimentation
├── mental_healt_analysis.py       # Script for automated training
├── mental_health_dataset.csv      # Raw survey data (anonymised)
├── mental_health_model.pkl        # Latest trained model (binary pickle)
└── README.md                      # You are here
```

