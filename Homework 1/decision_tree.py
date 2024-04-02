import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# load the data
data_train_path = 'complex_nonlinear_data-Sheet1.csv'
data_test_path = 'new_complex_nonlinear_data-Sheet1.csv' 
data_train = pd.read_csv(data_train_path)
data_test = pd.read_csv(data_test_path)

# prepare the data
X_train = data_train.iloc[:, 0].values.reshape(-1, 1)
y_train = data_train.iloc[:, 1].values
X_test = data_test.iloc[:, 0].values.reshape(-1, 1)
y_test = data_test.iloc[:, 1].valuess

# first train
model = DecisionTreeRegressor(max_depth=7)
model.fit(X_train, y_train)
predictions_train = model.predict(X_train)

# calculate and remove the outliers
residuals = np.abs(y_train - predictions_train)
threshold = np.std(residuals) * 1.75 
outliers = residuals > threshold
X_train_filtered = X_train[~outliers, :]
y_train_filtered = y_train[~outliers]

# retrain and test
model.fit(X_train_filtered, y_train_filtered)
predictions_test = model.predict(X_test)

# calculate MSE
mse_test = mean_squared_error(y_test, predictions_test)

# plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(X_test, predictions_test, color='orange', label='Predicted', alpha=0.5)
plt.title('Predicted vs Actual Values on Test Data')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

print(f"The Mean Squared Error (MSE) on the test data is: {mse_test}")
plt.show()

