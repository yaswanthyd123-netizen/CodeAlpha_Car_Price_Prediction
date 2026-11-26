# Task 3 — Car Price Prediction with Machine Learning

A regression workflow that predicts a car's selling price from brand, year, kilometres driven, horsepower, fuel type and transmission.

## Included data — please read
`car_data.csv` is a **synthetic demonstration dataset** generated for this project. It is not a market-price database and must not be represented as real listings. Replace it with the internship's assigned dataset for a real-world result; preserve the target column `selling_price` and update feature names in `main.py` if needed.

## What it does
- Checks and cleans the data.
- Creates car age from model year.
- One-hot encodes categorical columns using a scikit-learn pipeline.
- Trains Linear Regression and Random Forest models.
- Reports MAE, RMSE and R²; saves an actual-vs-predicted plot.

## Run
```bash
python -m pip install -r requirements.txt
python main.py
```
