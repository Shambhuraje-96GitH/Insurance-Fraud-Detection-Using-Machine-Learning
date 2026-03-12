import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load Data
df = pd.read_csv('insurance_claims.csv')

# 2. Simple Cleaning (Drop non-essential columns like Policy ID)
df = df.drop(['policy_number', 'insured_zip'], axis=1)

# 3. Convert Categorical to Numerical
df = pd.get_dummies(df)

# 4. Split Data
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
