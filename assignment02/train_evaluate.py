import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def train_and_evaluate():
    # Set styling
    sns.set_theme(style='whitegrid', font_scale=1.0)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'house_prices.csv')
    output_dir = base_dir
    artifact_dir = '/home/groot/.gemini/antigravity/brain/d6798a9b-ca11-459a-9355-8976b4b92ebf'

    df = pd.read_csv(data_path)
    feature_cols = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Age_Years', 'Distance_City_Center', 'Location_Score']
    target_col = 'Price'

    X = df[feature_cols]
    y = df[target_col]

    # 80:20 Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Dataset summary:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples: {len(X_test)} (20%)")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}\n")

    # Feature scaling for distance/gradient based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model initializations
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(max_depth=6, min_samples_split=10, random_state=42),
            'scaled': False
        },
        'Random Forest': {
            'model': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=5, random_state=42),
            'scaled': False
        },
        'Neural Network (MLP)': {
            'model': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1500, random_state=42),
            'scaled': True
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42),
            'scaled': False
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsRegressor(n_neighbors=7, weights='distance'),
            'scaled': True
        }
    }

    results = []
    predictions = {}

    print("--- Training and Evaluation ---")
    for name, config in models.items():
        clf = config['model']
        is_scaled = config['scaled']

        X_tr = X_train_scaled if is_scaled else X_train
        X_te = X_test_scaled if is_scaled else X_test

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Algorithm': name,
            'MAE ($k)': np.round(mae, 4),
            'MSE ($k^2)': np.round(mse, 4),
            'RMSE ($k)': np.round(rmse, 4),
            'R2 Score': np.round(r2, 4)
        })
        print(f"[{name}] MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)

    print("\n================ COMPARATIVE PERFORMANCE TABLE ================")
    print(metrics_df.to_string(index=False))
    print("===============================================================\n")

    # -------------------------------------------------------------
    # Visualization 1: Actual vs. Predicted (2x3 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    y_min, y_max = y_test.min() - 20, y_test.max() + 20

    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.55, edgecolors='k', linewidth=0.5, color='#1f77b4', s=35)
        # Reference diagonal line (ideal predictions)
        ax.plot([y_min, y_max], [y_min, y_max], color='#d62728', linestyle='--', linewidth=2.0, label='Ideal Fit (y = x)')
        
        # Trendline of actual vs predicted
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(y_test), p(np.sort(y_test)), color='#2ca02c', linestyle='-', linewidth=1.5, label='Regression Trend')

        r2_val = metrics_df.loc[metrics_df['Algorithm'] == name, 'R2 Score'].values[0]
        rmse_val = metrics_df.loc[metrics_df['Algorithm'] == name, 'RMSE ($k)'].values[0]

        ax.set_title(f"{name}\n$R^2$: {r2_val:.4f} | RMSE: {rmse_val:.2f}$k", fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual House Price ($k)', fontsize=10)
        ax.set_ylabel('Predicted House Price ($k)', fontsize=10)
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    act_vs_pred_file = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(act_vs_pred_file, dpi=300)
    plt.close()
    print(f"Saved {act_vs_pred_file}")

    # -------------------------------------------------------------
    # Visualization 2: Residuals Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True, ax=ax, color=palette[idx], stat="density", bins=25)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
        ax.set_title(f"Residuals: {name}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Residual Error (Actual - Predicted) [$k]', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    residuals_file = os.path.join(output_dir, 'residuals_distribution.png')
    plt.savefig(residuals_file, dpi=300)
    plt.close()
    print(f"Saved {residuals_file}")

    # -------------------------------------------------------------
    # Visualization 3: Metrics Comparison Bar Chart
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart 1: MAE and RMSE
    x_indices = np.arange(len(metrics_df))
    bar_width = 0.35

    ax1.bar(x_indices - bar_width/2, metrics_df['MAE ($k)'], width=bar_width, label='MAE ($k)', color='#3498db', edgecolor='black', linewidth=0.5)
    ax1.bar(x_indices + bar_width/2, metrics_df['RMSE ($k)'], width=bar_width, label='RMSE ($k)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(metrics_df['Algorithm'], rotation=25, ha='right', fontsize=10)
    ax1.set_ylabel('Error in Thousands of Dollars ($k)', fontsize=11)
    ax1.set_title('Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Bar chart 2: R^2 Score
    colors_r2 = ['#2ecc71' if v > 0.9 else '#f39c12' for v in metrics_df['R2 Score']]
    bars = ax2.bar(metrics_df['Algorithm'], metrics_df['R2 Score'], color=colors_r2, edgecolor='black', linewidth=0.5, width=0.55)
    ax2.set_xticks(range(len(metrics_df)))
    ax2.set_xticklabels(metrics_df['Algorithm'], rotation=25, ha='right', fontsize=10)
    ax2.set_ylabel('$R^2$ Score (Coefficient of Determination)', fontsize=11)
    ax2.set_title('R² Score Comparison Across Models', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    metrics_file = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(metrics_file, dpi=300)
    plt.close()
    print(f"Saved {metrics_file}")

    # Copy plots to artifact directory for markdown embedding if available
    if os.path.exists(artifact_dir):
        for img in ['actual_vs_predicted.png', 'residuals_distribution.png', 'metrics_comparison.png']:
            src = os.path.join(output_dir, img)
            dst = os.path.join(artifact_dir, img)
            shutil.copyfile(src, dst)
            print(f"Copied {img} to {dst}")

if __name__ == '__main__':
    train_and_evaluate()
