
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


df = sns.load_dataset("penguins")
df = df.dropna()


X = df[["bill_length_mm", "flipper_length_mm", "body_mass_g"]]
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "penguins_model.pkl")
print("Modelo salvo com sucesso.")
