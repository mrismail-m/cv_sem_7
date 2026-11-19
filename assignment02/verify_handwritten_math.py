import os
import numpy as np
import pandas as pd

def run_handwritten_verification():
    print("=================================================================")
    print("STEP-BY-STEP HANDWRITTEN VERIFICATION ON 5-POINT MINI DATASET")
    print("=================================================================\n")

    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'sample_5_points.csv'))
    print("Dataset:")
    print(df)
    
    X_raw = df[['Square_Feet_k', 'Age_Decades']].values
    y = df['Price'].values
    N = len(y)

    # -------------------------------------------------------------
    # 1. LINEAR REGRESSION (Ordinary Least Squares via Normal Eq)
    # -------------------------------------------------------------
    print("\n--- 1. LINEAR REGRESSION ---")
    X = np.column_stack([np.ones(N), X_raw])
    print("Design Matrix X (including intercept column of 1s):")
    print(X)

    XTX = X.T @ X
    XTy = X.T @ y
    print("\nX^T * X matrix:")
    print(XTX)
    print("\nX^T * y vector:")
    print(XTy)

    beta = np.linalg.inv(XTX) @ XTy
    print("\nAnalytical Coefficients beta = (X^T * X)^(-1) * X^T * y:")
    print(f"beta_0 (Intercept): {beta[0]:.4f}")
    print(f"beta_1 (Square_Feet_k): {beta[1]:.4f}")
    print(f"beta_2 (Age_Decades): {beta[2]:.4f}")

    y_pred_lr = X @ beta
    residuals_lr = y - y_pred_lr
    print("\nPredictions and Residuals:")
    for i in range(N):
        print(f"Point {i+1}: Actual={y[i]:.1f}, Pred={y_pred_lr[i]:.4f}, Residual={residuals_lr[i]:.4f}")
    mse_lr = np.mean(residuals_lr**2)
    print(f"Sample MSE = {mse_lr:.4f}")

    # Gradient Descent Step 1
    # Initialize w = [0, 0, 0], alpha = 0.01
    w_0 = np.zeros(3)
    # Loss J = (1/N) * sum (y - Xw)^2 -> grad = -(2/N) * X^T (y - Xw)
    grad_0 = -(2.0 / N) * (X.T @ (y - X @ w_0))
    alpha = 0.01
    w_1 = w_0 - alpha * grad_0
    print("\nGradient Descent (Step 1):")
    print(f"Initial w_0: {w_0}")
    print(f"Gradient at w_0: {grad_0}")
    print(f"Updated w_1 (alpha={alpha}): {w_1}")

    # -------------------------------------------------------------
    # 2. DECISION TREE REGRESSION (Variance Reduction)
    # -------------------------------------------------------------
    print("\n--- 2. DECISION TREE REGRESSION ---")
    y_mean_root = np.mean(y)
    mse_root = np.mean((y - y_mean_root)**2)
    print(f"Root Mean y = {y_mean_root:.2f}")
    print(f"Root MSE = {mse_root:.2f}")

    # Candidate splits on x1: 1.5, 2.5, 3.5
    # Candidate splits on x2: 1.5, 2.5
    splits = [
        ('x1 <= 1.5', X_raw[:, 0] <= 1.5),
        ('x1 <= 2.5', X_raw[:, 0] <= 2.5),
        ('x1 <= 3.5', X_raw[:, 0] <= 3.5),
        ('x2 <= 1.5', X_raw[:, 1] <= 1.5),
        ('x2 <= 2.5', X_raw[:, 1] <= 2.5)
    ]
    
    best_split = None
    best_gain = -1.0

    print("\nEvaluating Candidate Splits:")
    for name, mask in splits:
        y_L = y[mask]
        y_R = y[~mask]
        n_L, n_R = len(y_L), len(y_R)
        if n_L == 0 or n_R == 0:
            continue
        mse_L = np.mean((y_L - np.mean(y_L))**2)
        mse_R = np.mean((y_R - np.mean(y_R))**2)
        weighted_mse = (n_L / N) * mse_L + (n_R / N) * mse_R
        gain = mse_root - weighted_mse
        print(f"Split [{name}]: N_L={n_L}, Mean_L={np.mean(y_L):.2f}, MSE_L={mse_L:.2f} | N_R={n_R}, Mean_R={np.mean(y_R):.2f}, MSE_R={mse_R:.2f} | Weighted MSE={weighted_mse:.2f} | Gain={gain:.2f}")
        if gain > best_gain:
            best_gain = gain
            best_split = (name, mask, np.mean(y_L), np.mean(y_R), weighted_mse)

    print(f"\nOptimal Split: {best_split[0]} with Variance Reduction (Gain) = {best_gain:.2f}")
    print(f"Left Leaf Prediction: {best_split[2]:.2f}")
    print(f"Right Leaf Prediction: {best_split[3]:.2f}")

    # -------------------------------------------------------------
    # 3. RANDOM FOREST REGRESSION (Ensemble Bagging)
    # -------------------------------------------------------------
    print("\n--- 3. RANDOM FOREST REGRESSION ---")
    # Tree 1: Bootstrap sample indices [0, 1, 2, 2, 4], feature x1 (Square_Feet_k)
    boot1_idx = [0, 1, 2, 2, 4]
    X_b1 = X_raw[boot1_idx, 0] # only x1
    y_b1 = y[boot1_idx]
    # Stump on x1 <= 2.0 vs x1 > 2.0
    mask1 = X_b1 <= 2.0
    pred1_L = np.mean(y_b1[mask1])
    pred1_R = np.mean(y_b1[~mask1])
    print(f"Tree 1 (Bootstrap indices {boot1_idx}, Feature x1):")
    print(f"  Split: x1 <= 2.0 -> Left Mean = {pred1_L:.2f}, Right Mean = {pred1_R:.2f}")

    # Tree 2: Bootstrap sample indices [1, 2, 3, 4, 4], feature x2 (Age_Decades)
    boot2_idx = [1, 2, 3, 4, 4]
    X_b2 = X_raw[boot2_idx, 1] # only x2
    y_b2 = y[boot2_idx]
    # Stump on x2 <= 2.0 vs x2 > 2.0
    mask2 = X_b2 <= 2.0
    pred2_L = np.mean(y_b2[mask2])
    pred2_R = np.mean(y_b2[~mask2])
    print(f"Tree 2 (Bootstrap indices {boot2_idx}, Feature x2):")
    print(f"  Split: x2 <= 2.0 -> Left Mean = {pred2_L:.2f}, Right Mean = {pred2_R:.2f}")

    # Predict on test query point x* = [2.5, 1.5]
    x_star = np.array([2.5, 1.5])
    t1_pred = pred1_R if x_star[0] > 2.0 else pred1_L
    t2_pred = pred2_L if x_star[1] <= 2.0 else pred2_R
    rf_pred = 0.5 * (t1_pred + t2_pred)
    print(f"\nEnsemble Prediction for Query Point x* = [2.5, 1.5]:")
    print(f"  Tree 1 Prediction: {t1_pred:.2f}")
    print(f"  Tree 2 Prediction: {t2_pred:.2f}")
    print(f"  Random Forest Average: (1/2)*({t1_pred:.2f} + {t2_pred:.2f}) = {rf_pred:.2f}")

    # -------------------------------------------------------------
    # 4. NEURAL NETWORK REGRESSION (MLP 2-2-1 Forward & Backprop)
    # -------------------------------------------------------------
    print("\n--- 4. NEURAL NETWORK REGRESSION (MLP 2-2-1) ---")
    # Architecture: 2 inputs -> 2 hidden neurons (ReLU) -> 1 linear output
    # Sample point 1: x = [1.0, 1.0], target y = 120.0
    # To keep arithmetic clean and avoid huge gradients, we scale y by 1/100 (y_norm = 1.20)
    x_sample = np.array([1.0, 1.0])
    y_target = 1.20 # in hundreds

    # Fixed weights:
    W1 = np.array([[0.2, 0.4],
                   [-0.1, 0.3]]) # shape (2, 2): W1[0] for h1, W1[1] for h2
    b1 = np.array([0.1, -0.1])
    W2 = np.array([0.5, 0.8])    # shape (2,)
    b2 = 0.2

    print(f"Input x: {x_sample}, Target y_norm: {y_target}")
    print(f"Initial W1:\n{W1}\nInitial b1: {b1}")
    print(f"Initial W2: {W2}\nInitial b2: {b2}")

    # Forward Pass
    z1 = W1 @ x_sample + b1 # [0.2(1) + 0.4(1) + 0.1 = 0.7; -0.1(1) + 0.3(1) - 0.1 = 0.1]
    a1 = np.maximum(0, z1)  # ReLU
    y_hat = np.dot(W2, a1) + b2
    loss = 0.5 * (y_hat - y_target)**2
    print(f"\nForward Pass:")
    print(f"z1 = W1*x + b1 = {z1}")
    print(f"a1 = ReLU(z1) = {a1}")
    print(f"y_hat = W2*a1 + b2 = {y_hat:.4f}")
    print(f"Squared Error Loss L = 0.5 * (y_hat - y)^2 = {loss:.6f}")

    # Backpropagation
    d_loss_dyhat = y_hat - y_target # scalar
    grad_W2 = d_loss_dyhat * a1
    grad_b2 = d_loss_dyhat

    # Hidden layer gradients
    relu_grad = (z1 > 0).astype(float) # [1.0, 1.0]
    delta1 = (d_loss_dyhat * W2) * relu_grad # shape (2,)
    grad_W1 = np.outer(delta1, x_sample)
    grad_b1 = delta1

    print(f"\nBackpropagation Gradients:")
    print(f"dL/dy_hat: {d_loss_dyhat:.4f}")
    print(f"dL/dW2: {grad_W2}")
    print(f"dL/db2: {grad_b2:.4f}")
    print(f"delta1 (hidden layer error): {delta1}")
    print(f"dL/dW1:\n{grad_W1}")
    print(f"dL/db1: {grad_b1}")

    # SGD update with eta = 0.1
    eta = 0.1
    W2_new = W2 - eta * grad_W2
    b2_new = b2 - eta * grad_b2
    W1_new = W1 - eta * grad_W1
    b1_new = b1 - eta * grad_b1
    print(f"\nUpdated Weights (eta={eta}):")
    print(f"W2_new: {W2_new}")
    print(f"b2_new: {b2_new:.4f}")
    print(f"W1_new:\n{W1_new}")
    print(f"b1_new: {b1_new}")

    # -------------------------------------------------------------
    # 5. XGBOOST REGRESSION (Gradient & Hessian Splitting)
    # -------------------------------------------------------------
    print("\n--- 5. XGBOOST REGRESSION ---")
    # Base prediction y_hat^(0) = mean(y) = 220.0
    # Objective: L = sum 0.5 * (y_i - y_hat_i)^2
    # Gradient g_i = dL/dy_hat = y_hat_i - y_i
    # Hessian h_i = d^2L/dy_hat^2 = 1.0
    y_hat_0 = 220.0
    g = y_hat_0 - y
    h = np.ones(N)
    lam = 1.0 # L2 regularization parameter
    gamma = 0.0 # Tree complexity penalty

    print(f"Base model prediction y_hat^(0) = {y_hat_0}")
    print("Sample residuals and gradients:")
    for i in range(N):
        print(f"Point {i+1}: y={y[i]:.1f}, g_{i+1} = {g[i]:.1f}, h_{i+1} = {h[i]:.1f}")

    sim_root = (np.sum(g)**2) / (np.sum(h) + lam)
    print(f"\nRoot Similarity Score = (sum(g))^2 / (sum(h) + lambda) = ({np.sum(g)})^2 / ({np.sum(h)} + {lam}) = {sim_root:.4f}")

    # Candidate split: x1 <= 2.5 vs x1 > 2.5
    mask_xgb = X_raw[:, 0] <= 2.5
    g_L, h_L = g[mask_xgb], h[mask_xgb]
    g_R, h_R = g[~mask_xgb], h[~mask_xgb]

    sim_L = (np.sum(g_L)**2) / (np.sum(h_L) + lam)
    sim_R = (np.sum(g_R)**2) / (np.sum(h_R) + lam)
    gain_xgb = 0.5 * (sim_L + sim_R - sim_root) - gamma

    w_L = -np.sum(g_L) / (np.sum(h_L) + lam)
    w_R = -np.sum(g_R) / (np.sum(h_R) + lam)

    print(f"\nSplit Candidate: x1 <= 2.5")
    print(f"  Left Group (Points 1,2,3): sum(g_L) = {np.sum(g_L):.1f}, sum(h_L) = {np.sum(h_L):.1f}")
    print(f"  Left Similarity Score = {sim_L:.4f}")
    print(f"  Left Optimal Leaf Weight w_L* = -sum(g_L)/(sum(h_L) + lambda) = {w_L:.4f}")
    print(f"  Right Group (Points 4,5): sum(g_R) = {np.sum(g_R):.1f}, sum(h_R) = {np.sum(h_R):.1f}")
    print(f"  Right Similarity Score = {sim_R:.4f}")
    print(f"  Right Optimal Leaf Weight w_R* = -sum(g_R)/(sum(h_R) + lambda) = {w_R:.4f}")
    print(f"  XGBoost Split Gain = 0.5 * (Sim_L + Sim_R - Sim_Root) = {gain_xgb:.4f}")

    # Updated predictions for iteration 1 with eta = 0.3
    eta_xgb = 0.3
    print(f"\nUpdated Predictions (eta={eta_xgb}):")
    for i in range(N):
        w_assigned = w_L if mask_xgb[i] else w_R
        y_hat_1 = y_hat_0 + eta_xgb * w_assigned
        print(f"Point {i+1}: y={y[i]:.1f}, y_hat^(0)={y_hat_0:.1f}, w*={w_assigned:.4f}, y_hat^(1)={y_hat_1:.4f}")

    # -------------------------------------------------------------
    # 6. K-NEAREST NEIGHBORS REGRESSION (Distance Metrics & Averaging)
    # -------------------------------------------------------------
    print("\n--- 6. K-NEAREST NEIGHBORS REGRESSION ---")
    # Test query point x_q = [2.5, 2.0]
    x_q = np.array([2.5, 2.0])
    print(f"Query Point x_q: {x_q}")
    
    distances = []
    print("\nEuclidean Distances d(x_q, x_i) = sqrt((x_q1 - x_i1)^2 + (x_q2 - x_i2)^2):")
    for i in range(N):
        d = np.sqrt(np.sum((x_q - X_raw[i])**2))
        distances.append((i, d, y[i]))
        print(f"Point {i+1} ({X_raw[i,0]}, {X_raw[i,1]}): d = sqrt(({x_q[0]} - {X_raw[i,0]})^2 + ({x_q[1]} - {X_raw[i,1]})^2) = {d:.4f}, Actual y = {y[i]:.1f}")

    # Sort by distance
    distances.sort(key=lambda item: item[1])
    print("\nSorted Neighbors by Proximity:")
    for rank, (idx, dist, val) in enumerate(distances):
        print(f"Rank {rank+1}: Point {idx+1}, Distance = {dist:.4f}, Target y = {val:.1f}")

    # k = 2
    k2_neighbors = distances[:2]
    y_pred_k2 = np.mean([item[2] for item in k2_neighbors])
    print(f"\nUniform KNN (k=2): Average of Point {k2_neighbors[0][0]+1} and Point {k2_neighbors[1][0]+1}:")
    print(f"  y_hat(k=2) = ({k2_neighbors[0][2]} + {k2_neighbors[1][2]}) / 2 = {y_pred_k2:.2f}")

    # k = 3
    k3_neighbors = distances[:3]
    y_pred_k3 = np.mean([item[2] for item in k3_neighbors])
    print(f"\nUniform KNN (k=3): Average of Top 3 Points:")
    print(f"  y_hat(k=3) = ({k3_neighbors[0][2]} + {k3_neighbors[1][2]} + {k3_neighbors[2][2]}) / 3 = {y_pred_k3:.2f}")

    # Inverse Distance Weighted (k=2)
    w_knn = [1.0 / item[1] for item in k2_neighbors]
    y_pred_idw = np.sum([w_knn[j] * k2_neighbors[j][2] for j in range(2)]) / np.sum(w_knn)
    print(f"\nInverse Distance Weighted (k=2):")
    print(f"  Weight 1 = 1/{k2_neighbors[0][1]:.4f} = {w_knn[0]:.4f}")
    print(f"  Weight 2 = 1/{k2_neighbors[1][1]:.4f} = {w_knn[1]:.4f}")
    print(f"  y_hat_IDW = ({w_knn[0]:.4f}*{k2_neighbors[0][2]} + {w_knn[1]:.4f}*{k2_neighbors[1][2]}) / ({w_knn[0]:.4f} + {w_knn[1]:.4f}) = {y_pred_idw:.2f}")

if __name__ == '__main__':
    run_handwritten_verification()
