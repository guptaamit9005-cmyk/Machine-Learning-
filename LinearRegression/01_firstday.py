
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
reg.fit(X, y)
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
prediction = reg.predict([[3, 3]])
print("Prediction:", prediction)
