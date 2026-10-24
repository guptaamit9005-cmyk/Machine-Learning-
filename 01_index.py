
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = {
    "Area": [500, 700, 800, 1000, 1200,
             1400, 1600, 1800, 2000, 2200],
    "Price": [15, 20, 22, 28, 32,
              38, 42, 48, 52, 58]
}
df = pd.DataFrame(data)
print("House Price Dataset:")
print(df)
X = df[["Area"]]
y = df["Price"]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nActual Prices:", y_test.tolist())
print("Predicted Prices:", y_pred.round(2).tolist())
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))
area = 1500
predicted_price = model.predict(pd.DataFrame({"Area": [area]}))
print(f"\nHouse Area: {area} sq ft")
print(f"Predicted Price: ₹{predicted_price[0]:.2f} lakh")
plt.scatter(X["Area"], y, color="blue", label="Actual Prices")
plt.plot(X["Area"], model.predict(X), color="red",
         label="Regression Line")

plt.xlabel("House Area (sq ft)")
plt.ylabel("House Price (₹ lakh)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()
