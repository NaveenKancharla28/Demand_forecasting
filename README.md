# Demand_forecasting

Walmart Sales — Demand Forecasting (Random Forest & XGBoost)

Robust, production-ready pipeline to forecast weekly Walmart-style sales using tree-based ML models. The repo includes code, data prep, EDA, model training, evaluation, and visualizations.

Best model: XGBoost (MAE ≈ $929.45, RMSE ≈ $1,196.89, R² ≈ 0.919, MAPE ≈ 5.86%)
📌 Project Highlights

End-to-end script: load → clean → feature-engineer → train → evaluate → visualize

Models: Random Forest and XGBoost with cross-validation

Time-aware features (lags, rolling means, cyclical encodings for week/month)

Clear business insights + recommendations for inventory planning

Works out-of-the-box: if no dataset is found, the script generates a realistic Walmart-style sample dataset to demonstrate the pipeline

📂 Recommended Repo Structure
demand-forecasting/
├─ data/
│  └─ walmart_sales.csv        # (optional) real dataset; script auto-generates sample data if missing
├─ assets/
│  ├─ pred_vs_actual.png
│  ├─ scatter_rf_xgb.png
│  └─ eda_grid.png
├─ wallmart_sales.py           # main script (note: filename kept as-is)
├─ requirements.txt
└─ README.md


If you use different image names/paths, update the links in this README accordingly.

🧠 What’s Inside (Pipeline)

Load & Inspect

Looks for data/walmart_sales.csv.

If missing: “Dataset not found. Creating sample Walmart-style data…” and proceeds with generated data of the same schema.

Preprocessing & Feature Engineering

Converts Date to datetime and derives: Year, Month, Week, DayOfYear, cyclical encodings (Month_sin/cos, Week_sin/cos)

Creates lags (e.g., Sales_lag_1, Sales_lag_2) and rolling features (e.g., Sales_rolling_4)

One-hot/label encodes categorical features as needed

Modeling

Random Forest and XGBoost

K-fold cross-validation (reports mean ± std of MAE)

Final training on full training split + test evaluation

Evaluation & Plots

Error metrics: MAE, RMSE, R², MAPE

Plots: predictions vs actuals, predicted vs actual scatter, EDA (trends, distributions, holiday effect)

Feature importances (XGBoost)

🧪 Results (on provided run)

Dataset

Raw: (411,416, 9) → Final after FE: (407,301, 26)

Features used in modeling: 24

Training samples: 353,841

Exploratory Stats

Total Sales: $6,727,021,301.31

Average Weekly Sales: $16,516.09

Std Dev (Weekly): $4,099.66

Holiday Sales Boost: +21.4%

Cross-Validation (MAE)

Random Forest: $954.39 ± $8.19

XGBoost: $936.92 ± $7.41

Test Performance

Model	MAE ($)	RMSE ($)	R²	MAPE
Random Forest	945.71	1,217.89	0.916	5.97%
XGBoost	929.45	1,196.89	0.919	5.86%

Top 10 Features (XGBoost importance)

Month_sin (0.900)

Week_sin (0.044)

Month_cos (0.018)

DayOfYear (0.011)

Sales_rolling_4 (0.010)

Month (0.004)

Week_cos (0.003)

Year (0.003)

Sales_lag_2 (0.002)

Week (0.002)

📊 Visualizations

Add these images to assets/ (or update paths).

Left: Random Forest, Right: XGBoost — Predicted vs Actual (with R²)


EDA Grid: monthly trend, average sales by month, overall distribution, holiday vs non-holiday averages


🛠️ Setup (macOS + VS Code)
# clone
git clone <your-repo-url>.git
cd demand-forecasting

# create & activate venv (macOS)
python3 -m venv .venv
source .venv/bin/activate

# install deps
python -m pip install -U pip
pip install -r requirements.txt


requirements.txt (example)

pandas
numpy
scikit-learn
xgboost
matplotlib

▶️ Run
# from repo root
python3 wallmart_sales.py


Sample console output

============================================================
WALMART SALES DEMAND FORECASTING
============================================================

1. Loading and exploring the dataset...
📁 Dataset not found. Creating sample Walmart-style data...

Dataset shape: (411416, 9)
...
2. Data preprocessing and feature engineering...
✅ Data preprocessing completed
Final dataset shape: (407301, 26)

3. Exploratory Data Analysis...
Total Sales: $6,727,021,301.31
Average Weekly Sales: $16,516.09
Holiday Sales Boost: 21.4%

4. Building predictive models...
Random Forest CV MAE: $954.39 ± $8.19
XGBoost     CV MAE: $936.92 ± $7.41
...
5. Training final models...
Random Forest  → MAE: $945.71 | RMSE: $1,217.89 | R²: 0.916 | MAPE: 5.97%
XGBoost        → MAE: $929.45 | RMSE: $1,196.89 | R²: 0.919 | MAPE: 5.86%

📥 Using Your Own Data

Place a CSV at data/walmart_sales.csv with columns:

Store, Dept, Date, Weekly_Sales, IsHoliday, Temperature, Fuel_Price, CPI, Unemployment


Date must be parseable (e.g., YYYY-MM-DD).

The script will automatically use this file instead of generating sample data.

🧩 Business Insights

Best model: XGBoost — predictions are, on average, within ~$929 of actual weekly sales

Holiday uplift: ~21.4% sales increase on holidays → stock up + schedule staffing accordingly

Seasonality matters: cyclical month/week encodings dominate importance → align promos with seasonal peaks

Process suggestion: retrain monthly, monitor drift, and incorporate exogenous signals (weather, fuel, CPI, unemployment)

Next steps

Batch/online inference endpoint (FastAPI)

Scheduled retraining (cron/GitHub Actions)

Integration with inventory planning dashboards

⚠️ Notes & Limitations

The auto-generated dataset is for demonstration when a real dataset is absent.

Results above reflect one train/test split; performance can vary with data recency and store/department mix.

For production, add rigorous backtesting (e.g., rolling-origin) and feature stability checks.

🏷️ Acknowledgments

Schema inspired by the Kaggle “Walmart Recruiting — Store Sales Forecasting” challenge.

📜 License

MIT — feel free to use, modify, and share with attribution.

🙌 Author

Built on macOS with VS Code. If this helps you, please ⭐ the repo!
