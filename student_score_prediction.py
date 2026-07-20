import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Get absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

file_path = os.path.join(project_root, "data", "student_scores.csv")

print("Dataset path:", file_path)

df = pd.read_csv(file_path)

X = df[['StudyHours', 'SleepHours', 'PreviousScore']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

new_data = pd.DataFrame({
    'StudyHours':[6],
    'SleepHours':[7],
    'PreviousScore':[65]
})

prediction = model.predict(new_data)
print("Predicted Score:", prediction[0])

"""prediction = model.predict([[6,7,65]])
print("Predicted Score:", prediction[0])"""

mae = mean_absolute_error(y_test, model.predict(X_test))
print("Mean Absolute Error:", mae)
