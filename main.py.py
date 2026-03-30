import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
data = pd.read_csv("students.csv")

print(data.head())
print(data.info())  # columns, data types, missing values
print(data.describe())  #stats (mean, min, max etc.)
data.plot(x='study_hours', y='overall_score', kind='scatter')
plt.show()
X = data[['study_hours']]   # input
y = data['overall_score']     # output
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions=np.array(predictions)
predictions=(np.clip(predictions,0,100))
final=float(predictions[0])
print(final)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(results)
plt.figure()

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, predictions, color='red', label='Predicted')

plt.legend()
plt.xlabel("Input")
plt.ylabel("Output")

plt.show()
# Take user input
hours = float(input("Enter study hours: "))

# Convert into proper format
new_data = [[hours]]
# Predict
prediction = model.predict(new_data)
prediction=np.clip(prediction,0,100)
prediction = float(prediction[0])






# Show result
print("Predicted Marks:", prediction)
