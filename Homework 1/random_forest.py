import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# load data
data_train_path = 'complex_nonlinear_data-Sheet1.csv'  # 训练集路径
data_test_path = 'new_complex_nonlinear_data-Sheet1.csv'  # 测试集路径
data_train = pd.read_csv(data_train_path)
data_test = pd.read_csv(data_test_path)

# prepare the data
X_train = data_train.iloc[:, 0].values.reshape(-1, 1)  # 假设特征在第一列
y_train = data_train.iloc[:, 1].values  # 假设目标变量在第二列
X_test = data_test.iloc[:, 0].values.reshape(-1, 1)
y_test = data_test.iloc[:, 1].values

# train the data
model = RandomForestRegressor(n_estimators=225, max_depth=7, min_samples_leaf=2, min_samples_split=2)
model.fit(X_train, y_train)
predictions_train = model.predict(X_train)

# find the outliers
residuals = np.abs(y_train - predictions_train)
threshold = np.std(residuals) * 3.9  
outliers = residuals > threshold
X_train_filtered = X_train[~outliers, :]
y_train_filtered = y_train[~outliers]

model.fit(X_train_filtered, y_train_filtered)
predictions_test = model.predict(X_test)

mse_test = mean_squared_error(y_test, predictions_test)

# plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(X_test, predictions_test, color='orange', label='Predicted', alpha=0.5)
plt.title('Predicted vs Actual Values on Test Data (Random Forest)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

# output the mse
print(f"The Mean Squared Error (MSE) on the test data is: {mse_test}")
plt.show()

