import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import norm

# -------------------------------
# LOAD DATASET (YOUR FILE)
# -------------------------------
df = pd.read_csv("plant_disease_dataset.csv")  # <-- replace with your file name

print("\nDataset Preview:")
print(df.head())

# -------------------------------
# 1. DATA CLEANING (Exp 7)
# -------------------------------
print("\n--- Data Cleaning ---")
df = df.select_dtypes(include=[np.number])  # keep numeric only
df.fillna(df.mean(), inplace=True)

print("Cleaned Data:")
print(df.head())

# -------------------------------
# 2. CORRELATION (Exp 1)
# -------------------------------
print("\n--- Correlation Matrix ---")
corr = df.corr()
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 3. SAMPLING (Exp 4)
# -------------------------------
print("\n--- Random Sampling (25%) ---")
sample = df.sample(frac=0.25)
print(sample)

# -------------------------------
# 4. NUMPY OPERATIONS (Exp 6)
# -------------------------------
print("\n--- NumPy Operations ---")
arr = df.to_numpy()

print("Mean:", np.mean(arr))
print("Sum:", np.sum(arr))
print("Std Dev:", np.std(arr))

# -------------------------------
# 5. Z-TEST (Exp 5)
# -------------------------------
print("\n--- Z-Test ---")

sample_data = np.random.normal(150, 10, 40)
mean = np.mean(sample_data)
std = np.std(sample_data)

z = (mean - 140) / (std / np.sqrt(40))
critical = 1.64

print("Z-score:", z)

if z > critical:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

# -------------------------------
# 6. LINEAR REGRESSION (Exp 3)
# -------------------------------
print("\n--- Linear Regression ---")

if df.shape[1] >= 2:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict(X)

    plt.scatter(y, pred)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

# -------------------------------
# 7. LOGISTIC REGRESSION (Exp 2)
# -------------------------------
print("\n--- Logistic Regression ---")

# Convert target to binary (if needed)
if df.iloc[:, -1].nunique() > 2:
    y_bin = (df.iloc[:, -1] > df.iloc[:, -1].mean()).astype(int)
else:
    y_bin = df.iloc[:, -1]

X = df.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# -------------------------------
# 8. STUDENT ANALYSIS (Exp 9)
# -------------------------------
print("\n--- Student Analysis ---")

print("Average per column:")
print(df.mean())

print("Highest row index:", df.mean(axis=1).idxmax())
print("Lowest row index:", df.mean(axis=1).idxmin())

# -------------------------------
# 9. SALES / TREND (Exp 10)
# -------------------------------
print("\n--- Trend Visualization ---")

df.mean().plot(kind="bar")
plt.title("Column Averages")
plt.show()

# -------------------------------
# 10. BASIC FINANCE SIMULATION (Exp 8)
# -------------------------------
print("\n--- Finance Simulation ---")

returns = np.random.normal(0.001, 0.01, 100)
price = 100

prices = [price]

for r in returns:
    price = price * (1 + r)
    prices.append(price)

plt.plot(prices)
plt.title("Simulated Stock Price")
plt.show()

print("\n--- ALL EXPERIMENTS EXECUTED SUCCESSFULLY ---")