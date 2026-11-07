
# Step 1: Import libraries
import numpy as np
from sklearn.linear_model import Ridge

# Step 2: Create input data
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

# Step 3: Create output data
y = np.array([2, 4, 5, 4, 5])

# Step 4: Create Ridge Regression model
ridge = Ridge(alpha=1.0)

# Step 5: Train the model
ridge.fit(X, y)

# Step 6: Print coefficient
print("Coefficient:", ridge.coef_)

# Step 7: Print intercept
print("Intercept:", ridge.intercept_)

# Step 8: Make prediction
prediction = ridge.predict([[6]])

print("Prediction:", prediction)
