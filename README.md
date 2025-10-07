# Car Price Prediction (Mini Project)

This Streamlit app predicts used car selling prices using basic machine learning.

## Quick start

1. Install dependencies:

```powershell
& 'C:\Program Files\Python313\python.exe' -m pip install -r requirements.txt
```

2. Run the app:

```powershell
& 'C:\Program Files\Python313\python.exe' -m streamlit run 'c:\Users\sidha\car-price-prediction\app.py'
```

## Notes
- The app looks for dataset files in `car_data/` and the repo root. If no dataset is found you can upload a CSV via the app UI.
- The trained model (joblib) will be saved to `models/best_model.joblib` after training.

## Next steps
- Add hyperparameter tuning and model explainability (SHAP), improve feature engineering, and deploy the app.
