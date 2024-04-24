import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
#inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]
#1.8
x_train_with_bias = np.c_[np.ones(len(x_train)), x_train]
#1.13
theta_best = np.dot(np.matmul(np.linalg.inv(np.matmul(x_train_with_bias.T, x_train_with_bias)), x_train_with_bias.T), y_train)
print("Theta ", theta_best)
# TODO: calculate error
x_test_with_bias = np.c_[np.ones(len(x_test)), x_test]
#1.3
mse = np.mean(np.square(np.matmul(x_test_with_bias,theta_best) - y_test))
print("Error for closed-form solution ", mse)


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
##
# TODO: standardization
#1.15
x_train_norm = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)

x_train_with_bias_norm = np.c_[np.ones(len(x_train_norm)), x_train_norm]

# TODO: calculate theta using Batch Gradient Descent
y_train_norm = np.c_[y_train_norm]
theta = np.random.rand(2, 1)
learning_rate = 0.01
max_iter = 1000
for iteration in range(max_iter):
    #1.7
    gradients = 2/len(x_train_norm) * np.matmul(x_train_with_bias_norm.T, (np.matmul(x_train_with_bias_norm, theta) - y_train_norm))
    theta = theta - learning_rate * gradients
    np.matmul(x_train_with_bias_norm, theta)
    if iteration % 20 == 0:
        mse = np.mean(np.square(np.matmul(x_train_with_bias_norm,theta) - y_train_norm))
        print("Number of iteration ", iteration, "error: ", mse)

# TODO: calculate error
x_test_norm = (x_test - np.mean(x_train)) / np.std(x_train)
y_test_norm = (y_test - np.mean(y_train)) / np.std(y_train)
y_test_norm_pred = x_test_norm * theta[1] + theta[0]
y_pred = y_test_norm_pred * np.std(y_train) + np.mean(y_train)
mse = np.mean(np.square(y_test - y_pred))
print("Error for closed-form solution ", mse)



# plot the regression line
x = np.linspace(min(x_test_norm), max(x_test_norm), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test_norm, y_test_norm)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()