import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
# Decision Tree Feature Selection
def feature_selection():

       df = pd.read_csv('databases/SSISampledAfterKNNImputation.csv')
       df.drop(columns=df.columns[0], axis=1, inplace=True)
       X = df.drop(columns='SSI', axis=1)
       Y = df['SSI']
       X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

       # Build Decision Tree classifier to use in feature selection
       classifier = DecisionTreeClassifier()

       # Build feature selection object
       sfm = SelectFromModel(classifier, max_features=10)

       # Perform feature selection
       sfm.fit(X_train, y_train)

       # Which features were selected?
       selected_features = X_train.columns[sfm.get_support()]
       return selected_features
def train_XG(train_X, val_X, train_y, val_y):
       # fit model no training data
       xg = XGBClassifier()
       xg.fit(train_X, train_y)
       y_pred = xg.predict(val_X)
       # Model Evaluation metrics 
      
       print('Accuracy Score : '  + str(accuracy_score(val_y,y_pred)))
       accuracy = accuracy_score(val_y,y_pred)
       print('Precision Score : ' + str(precision_score(val_y,y_pred)))
       print('Recall Score : '    + str(recall_score(val_y,y_pred)))
       print('F1 Score : '        + str(f1_score(val_y,y_pred)))
       print('AUC Score : '        + str(roc_auc_score(val_y,y_pred)))
       #Logistic Regression Classifier Confusion matrix
       from sklearn.metrics import confusion_matrix
       print('Confusion Matrix : \n' + str(confusion_matrix(val_y,y_pred)))
def main():
       df = pd.read_csv('databases/SSISampledAfterKNNImputation.csv')
       df.drop(columns=df.columns[0], axis=1, inplace=True)
       X = df.drop(columns='SSI', axis=1)
       Y = df['SSI']
       

       selected_features = list(feature_selection())
       print(selected_features)
       X2 =X[selected_features]
       train_X, val_X, train_y, val_y = train_test_split(X2.values, Y.values, test_size=0.3, random_state=109)
       train_XG(train_X, val_X, train_y, val_y)
if __name__ == "__main__":
    main()