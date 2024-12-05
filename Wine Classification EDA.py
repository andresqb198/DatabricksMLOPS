# Databricks notebook source
# MAGIC %md
# MAGIC # Wine Classification EDA

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df.head()


# COMMAND ----------

df.info()
df.describe()


# COMMAND ----------

df.isnull().sum()


# COMMAND ----------

plt.figure(figsize=(15,10))
sns.boxplot(data=df)


# COMMAND ----------

df[0].value_counts()


# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(0, axis=1))


# COMMAND ----------

X = X_scaled
y = df[0] - 1  


# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)


# COMMAND ----------

model.fit(X_train, y_train)


# COMMAND ----------

y_pred = model.predict(X_test)


# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# COMMAND ----------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


# COMMAND ----------

print(classification_report(y_test, y_pred))


# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# COMMAND ----------

import mlflow
import mlflow.sklearn

# COMMAND ----------

with mlflow.start_run():


    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)


    mlflow.sklearn.log_model(model, "random_forest_model")


# COMMAND ----------


