import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # compute metrics
    cf_mae = mean_absolute_error(y_test[:,0], y_pred[:,0])
    cf_rmse = mean_squared_error(y_test[:,0], y_pred[:,0], squared=False)
    # For SF and CR, round to nearest valid values
    sf_pred = np.round(y_pred[:,2]).astype(int)
    sf_true = np.round(y_test[:,2]).astype(int)
    sf_acc = accuracy_score(sf_true, sf_pred)
    # similar for CR
    cr_pred = np.round(y_pred[:,1]).astype(int)
    cr_true = np.round(y_test[:,1]).astype(int)
    cr_acc = accuracy_score(cr_true, cr_pred)
    return model, (cf_mae, cf_rmse, sf_acc, cr_acc)