# Walmart Sales Demand Forecasting
# A comprehensive machine learning approach to predict future sales

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Statistical forecasting
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("=" * 60)
print("WALMART SALES DEMAND FORECASTING")
print("=" * 60)

# 1. DATA LOADING AND EXPLORATION
print("\n1. Loading and exploring the dataset...")

def load_data():
    """Load Walmart sales data - adjust file paths as needed"""
    try:
        # Try different common file names for Walmart dataset
        possible_files = ['walmart_sales.csv', 'train.csv', 'sales.csv', 'walmart.csv','walmart-sales-dataset-of-45stores']
        
        for file in possible_files:
            try:
                df = pd.read_csv(file)
                print(f"✅ Successfully loaded: {file}")
                break
            except FileNotFoundError:
                continue
        else:
            # Create sample data if file not found
            print("📁 Dataset not found. Creating sample Walmart-style data...")
            df = create_sample_data()
            
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create realistic sample Walmart sales data"""
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(start='2010-02-05', end='2012-10-26', freq='W')
    
    # Create sample data
    stores = range(1, 46)  # 45 stores
    departments = range(1, 100)  # 99 departments
    
    data = []
    for date in dates:
        for store in stores:
            # Seasonal and trend effects
            week_of_year = date.week
            year = date.year
            
            # Holiday effects
            is_holiday = 1 if date.month == 12 or (date.month == 11 and date.day > 20) else 0
            
            for dept in np.random.choice(departments, size=np.random.randint(50, 80)):
                # Base sales with seasonality and noise
                base_sales = 15000 + 5000 * np.sin(2 * np.pi * week_of_year / 52)
                trend = (year - 2010) * 1000
                holiday_boost = is_holiday * np.random.uniform(0.2, 0.8) * base_sales
                noise = np.random.normal(0, base_sales * 0.1)
                
                weekly_sales = max(0, base_sales + trend + holiday_boost + noise)
                
                data.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': date,
                    'Weekly_Sales': weekly_sales,
                    'IsHoliday': is_holiday,
                    'Temperature': np.random.normal(70, 20),
                    'Fuel_Price': np.random.normal(3.5, 0.5),
                    'CPI': np.random.normal(200, 10),
                    'Unemployment': np.random.normal(7, 2)
                })
    
    return pd.DataFrame(data)

# Load the data
df = load_data()

# Display basic information
print(f"\nDataset shape: {df.shape}")
print("\nColumn names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

# 2. DATA PREPROCESSING AND FEATURE ENGINEERING
print("\n2. Data preprocessing and feature engineering...")

def preprocess_data(df):
    """Clean and prepare the data"""
    
    # Convert date column (handle different possible date column names)
    date_cols = ['Date', 'date', 'DATE', 'Week', 'week']
    date_col = None
    
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
    else:
        print("⚠️ No date column found, using index")
        df['Date'] = pd.date_range(start='2010-02-05', periods=len(df), freq='W')
    
    # Ensure we have the right column names
    if 'Weekly_Sales' not in df.columns:
        sales_cols = ['Weekly_Sales', 'sales', 'Sales', 'Weekly_Sales']
        for col in sales_cols:
            if col in df.columns:
                df['Weekly_Sales'] = df[col]
                break
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Remove outliers (sales < 0 or extremely high)
    df = df[df['Weekly_Sales'] >= 0]
    df = df[df['Weekly_Sales'] <= df['Weekly_Sales'].quantile(0.99)]
    
    return df

def create_features(df):
    """Create time-based and lag features"""
    
    # Ensure Date column exists
    if 'Date' not in df.columns:
        df['Date'] = pd.date_range(start='2010-02-05', periods=len(df), freq='W')
    
    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Cyclical encoding for seasonality
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    
    # Sort by Store, Dept, Date for lag features
    if 'Store' in df.columns and 'Dept' in df.columns:
        df = df.sort_values(['Store', 'Dept', 'Date'])
        
        # Lag features (previous weeks' sales)
        for lag in [1, 2, 4, 8, 12]:
            df[f'Sales_lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
        
        # Rolling averages
        for window in [4, 8, 12]:
            df[f'Sales_rolling_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # Holiday indicator (if not present)
    if 'IsHoliday' not in df.columns:
        # Create holiday indicator based on dates
        df['IsHoliday'] = ((df['Month'] == 12) | 
                          ((df['Month'] == 11) & (df['Date'].dt.day > 20))).astype(int)
    
    return df

# Preprocess the data
df = preprocess_data(df)
df = create_features(df)

print("✅ Data preprocessing completed")
print(f"Final dataset shape: {df.shape}")

# 3. EXPLORATORY DATA ANALYSIS
print("\n3. Exploratory Data Analysis...")

def plot_sales_analysis(df):
    """Create visualizations for sales analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sales over time
    if 'Store' in df.columns:
        monthly_sales = df.groupby([df['Date'].dt.to_period('M')])['Weekly_Sales'].sum()
    else:
        monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Weekly_Sales'].sum()
    
    axes[0, 0].plot(monthly_sales.index.astype(str), monthly_sales.values)
    axes[0, 0].set_title('Monthly Sales Trend')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Sales by month (seasonality)
    monthly_avg = df.groupby('Month')['Weekly_Sales'].mean()
    axes[0, 1].bar(monthly_avg.index, monthly_avg.values)
    axes[0, 1].set_title('Average Sales by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average Sales')
    
    # Sales distribution
    axes[1, 0].hist(df['Weekly_Sales'], bins=50, alpha=0.7)
    axes[1, 0].set_title('Sales Distribution')
    axes[1, 0].set_xlabel('Weekly Sales')
    axes[1, 0].set_ylabel('Frequency')
    
    # Holiday vs Non-Holiday sales
    if 'IsHoliday' in df.columns:
        holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
        axes[1, 1].bar(['Non-Holiday', 'Holiday'], holiday_sales.values)
        axes[1, 1].set_title('Average Sales: Holiday vs Non-Holiday')
        axes[1, 1].set_ylabel('Average Sales')
    
    plt.tight_layout()
    plt.show()

# Create visualizations
plot_sales_analysis(df)

# Print key statistics
print("\n📊 Key Statistics:")
print(f"Total Sales: ${df['Weekly_Sales'].sum():,.2f}")
print(f"Average Weekly Sales: ${df['Weekly_Sales'].mean():,.2f}")
print(f"Sales Standard Deviation: ${df['Weekly_Sales'].std():,.2f}")

if 'IsHoliday' in df.columns:
    holiday_avg = df[df['IsHoliday'] == 1]['Weekly_Sales'].mean()
    normal_avg = df[df['IsHoliday'] == 0]['Weekly_Sales'].mean()
    print(f"Holiday Sales Boost: {((holiday_avg / normal_avg - 1) * 100):.1f}%")

# 4. MODEL BUILDING
print("\n4. Building predictive models...")

def prepare_model_data(df):
    """Prepare data for machine learning models"""
    
    # Select features for modeling
    feature_cols = ['Year', 'Month', 'Week', 'Quarter', 'DayOfYear', 
                   'Month_sin', 'Month_cos', 'Week_sin', 'Week_cos']
    
    # Add available lag and rolling features
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    feature_cols.extend(lag_cols)
    
    # Add other available features
    other_cols = ['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Store', 'Dept']
    for col in other_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    # Remove rows with NaN values (due to lag features)
    model_df = df[feature_cols + ['Weekly_Sales']].dropna()
    
    X = model_df[feature_cols]
    y = model_df['Weekly_Sales']
    
    return X, y, feature_cols

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.3f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# Prepare data for modeling
X, y, feature_cols = prepare_model_data(df)

print(f"Features used: {len(feature_cols)}")
print(f"Training samples: {len(X)}")

# Time series split for validation
tscv = TimeSeriesSplit(n_splits=3)

# Model 1: Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation scores
rf_scores = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)
    rf_scores.append(mean_absolute_error(y_val, y_pred))

print(f"Random Forest CV MAE: ${np.mean(rf_scores):,.2f} ± ${np.std(rf_scores):,.2f}")

# Model 2: XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

xgb_scores = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_val)
    xgb_scores.append(mean_absolute_error(y_val, y_pred))

print(f"XGBoost CV MAE: ${np.mean(xgb_scores):,.2f} ± ${np.std(xgb_scores):,.2f}")

# Train final models on full dataset
print("\n5. Training final models...")

# Split data chronologically
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Evaluate models
rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
xgb_metrics = evaluate_model(y_test, xgb_pred, "XGBoost")

# Feature importance
print("\n📈 Top 10 Most Important Features (XGBoost):")
feature_importance = sorted(zip(feature_cols, xgb_model.feature_importances_), 
                           key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(feature_importance[:10]):
    print(f"  {i+1:2d}. {feature:<20}: {importance:.3f}")

# 6. VISUALIZATION OF RESULTS
print("\n6. Visualizing results...")

# Plot predictions vs actual
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Random Forest
axes[0].scatter(y_test, rf_pred, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')
axes[0].set_title(f'Random Forest Predictions\nR² = {rf_metrics["R2"]:.3f}')

# XGBoost
axes[1].scatter(y_test, xgb_pred, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Sales')
axes[1].set_ylabel('Predicted Sales')
axes[1].set_title(f'XGBoost Predictions\nR² = {xgb_metrics["R2"]:.3f}')

plt.tight_layout()
plt.show()

# Time series plot of predictions
plt.figure(figsize=(15, 6))
test_indices = range(len(y_test))
plt.plot(test_indices[:100], y_test.iloc[:100], label='Actual', linewidth=2)
plt.plot(test_indices[:100], xgb_pred[:100], label='XGBoost Predicted', linewidth=2)
plt.plot(test_indices[:100], rf_pred[:100], label='Random Forest Predicted', linewidth=2)
plt.xlabel('Time Period')
plt.ylabel('Weekly Sales')
plt.title('Sales Predictions vs Actual (First 100 Test Points)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. BUSINESS INSIGHTS AND RECOMMENDATIONS
print("\n" + "="*60)
print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("="*60)

best_model = "XGBoost" if xgb_metrics['MAE'] < rf_metrics['MAE'] else "Random Forest"
best_mae = min(xgb_metrics['MAE'], rf_metrics['MAE'])

print(f"\n🎯 Best Performing Model: {best_model}")
print(f"   Mean Absolute Error: ${best_mae:,.2f}")
print(f"   This means predictions are on average within ${best_mae:,.2f} of actual sales")

print(f"\n💡 Key Insights:")
print(f"   • Average weekly sales: ${y.mean():,.2f}")
print(f"   • Model accuracy represents {(1 - best_mae/y.mean())*100:.1f}% accuracy")

if 'IsHoliday' in df.columns:
    holiday_impact = df[df['IsHoliday']==1]['Weekly_Sales'].mean() / df[df['IsHoliday']==0]['Weekly_Sales'].mean()
    print(f"   • Holiday weeks show {(holiday_impact-1)*100:.1f}% higher sales")

print(f"\n🚀 Recommendations:")
print(f"   • Use {best_model} model for demand forecasting")
print(f"   • Focus inventory planning on top features identified")
print(f"   • Monitor model performance and retrain monthly")
print(f"   • Consider external factors (weather, economy) for better accuracy")

print(f"\n✅ Project completed successfully!")
print(f"   Next steps: Deploy model, set up automated retraining, integrate with inventory system")