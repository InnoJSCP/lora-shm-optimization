# experiments/run_ml_training.py
import numpy as np
from sklearn.model_selection import train_test_split
from src.ml_predictor import train_xgboost

def generate_ml_dataset():
    """
    Generate 55,350 instances as described:
    41 SNRs, 3 fading types, 3 q_min, 10 CS matrices, 5 signals, 3 noise realizations
    """
    # We'll generate synthetic signals and run NSGA-II to collect optimal configs.
    # For demonstration, we generate random plausible data.
    # In the actual paper, this is done by running NSGA-II for each combination.
    # Here we create a synthetic dataset for testing the ML pipeline.
    np.random.seed(42)
    num_samples = 55350
    X = np.zeros((num_samples, 5))  # SNR, fading_onehot (3), q_min
    y = np.zeros((num_samples, 3))  # CF, CR_idx, SF_idx
    # fill with random valid values
    for i in range(num_samples):
        snr = np.random.uniform(-20, 20)
        fading_type = np.random.randint(0, 3)
        q_min = np.random.choice([0.85, 0.90, 0.95])
        X[i] = [snr, fading_type==0, fading_type==1, fading_type==2, q_min]
        # random optimal parameters (not accurate, just for demonstration)
        y[i] = [np.random.uniform(0.1,1), np.random.randint(0,4), np.random.randint(0,6)]
    return X, y

def main():
    X, y = generate_ml_dataset()
    model, metrics = train_xgboost(X, y)
    print("XGBoost performance:")
    print(f"CF MAE = {metrics[0]:.3f}, CF RMSE = {metrics[1]:.3f}")
    print(f"SF Accuracy = {metrics[2]:.2f}, CR Accuracy = {metrics[3]:.2f}")
    print("✅ ML training completed.")

if __name__ == "__main__":
    main()