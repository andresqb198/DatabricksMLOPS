from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
import seaborn as sns
import numpy as np

class Trainer:

    def __init__(self, df):

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.drop(0, axis=1))
        

        self.X = X_scaled
        self.y = (df[0] - 1).astype(int)  

        self.model = None
        self.model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.y_pred = None

    def split_data(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(
            self.X, 
            self.y, 
            test_size=test_size, 
            random_state=random_state
        )
    
    def add_model(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def run_mlflow(self):
        with mlflow.start_run():

            self.model.fit(self.X_train, self.y_train)
            

            self.y_pred = self.model.predict(self.X_test)
            

            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred, average='weighted')
            recall = recall_score(self.y_test, self.y_pred, average='weighted')
            f1 = f1_score(self.y_test, self.y_pred, average='weighted')

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(self.model, "model")


            print(classification_report(self.y_test, self.y_pred))

    def draw_confusion_matrix(self):

        self.y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
