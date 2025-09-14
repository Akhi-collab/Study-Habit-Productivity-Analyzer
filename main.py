# main.py
# Study Habit Productivity Analyzer

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Step 1: Load CSV ---
df = pd.read_csv("study.csv")
print("\n📊 First 5 rows of data:")
print(df.head())

print("\n📈 Basic Stats:")
print(df.describe())

# --- Step 2: Visualizations ---
plt.figure(figsize=(6,4))
plt.scatter(df['study_hours'], df['productivity_score'], color='blue')
plt.xlabel('Study Hours')
plt.ylabel('Productivity Score')
plt.title('Study Hours vs Productivity')
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['breaks'], df['productivity_score'], color='green')
plt.xlabel('Breaks')
plt.ylabel('Productivity Score')
plt.title('Breaks vs Productivity')
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['mood'], df['productivity_score'], color='purple')
plt.xlabel('Mood')
plt.ylabel('Productivity Score')
plt.title('Mood vs Productivity')
plt.show()

# --- Step 3: Train Model ---
X = df[['study_hours','breaks','mood']]
y = df['productivity_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model trained!")
print("R² Score (accuracy):", round(r2, 3))

# --- Step 4: Try Prediction ---
print("\n🔮 Example prediction:")
example = pd.DataFrame([[5,2,5]], columns=['study_hours','breaks','mood'])
pred = model.predict(example)[0]
print("Predicted productivity score for 4 hrs, 1 break, mood 4 =", round(pred,2))