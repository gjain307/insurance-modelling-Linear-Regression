import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import argparse
import pandas as pd
import os

from sklearn import tree
from sklearn.externals import joblib
from joblib import dump, load

if __name__ == '__main__':
    
    
    df = pd.read_csv('./dataset/insurance_sample.csv');
    df_copy = pd.read_csv('./dataset/insurance_sample.csv');

    predictor = ['id', '_user_id', 'gender', 'driver_age', 'state', 'zip_code',
           'vehicle_make', 'vehicle_cost_new', 'vehicle_year', 'vehicle_ownership',
           'home_ownership', 'prior_carrier', 'prior_liability_limit',
           'first_name', 'marital_status', 'vehicle_model',
           'years_with_prior_carrier', 'years_licensed', 'driver_count',
           'vehicle_count', 'version', 'high', 'low', 'prov1high', 'prov1low',
           'prov1name', 'prov2high', 'prov2low', 'prov2name', 'prov3high',
           'prov3low', 'prov3name', 'prov4high', 'prov4low', 'prov4name',
           'prov5high', 'prov5low', 'prov5name']
    target_variable = ['last_name']


    df['last_name'].value_counts()


    df.apply(lambda x: sum(x.isnull()),axis=0)


    colums_to_remove = []
    for col in df.columns:
        if((len(df[col].unique()) == 1)):
            colums_to_remove.append(col)


    # Relationship between `vehicle_model` and `vehicle_ownership` is quite strange. Now we will look into if these columns are identical.

    # In[16]:


    colums_to_remove.append('version'); #Because version is different at only one row and all other rows have same values
    colums_to_remove.append('vehicle_model'); #Because vehicle model and vehicle ownership capture same information for all the records, so we will be keeping only one.
    # ID columns have unique value for every row so they won't be  useful to us.
    colums_to_remove.append('id'); 
    colums_to_remove.append('_user_id');
    colums_to_remove.append('high');
    colums_to_remove.append('low');
    colums_to_remove.append('prov1high');
    colums_to_remove.append('prov1low');
    colums_to_remove.append('prov2high');
    colums_to_remove.append('prov2low');
    colums_to_remove.append('prov3high');
    colums_to_remove.append('prov3low');
    colums_to_remove.append('prov4high');
    colums_to_remove.append('prov4low');
    colums_to_remove.append('prov5high');
    colums_to_remove.append('prov5low');




    predictor = [x for x in predictor if x not in colums_to_remove]


    categorical_variables = ['state', 'zip_code', 'vehicle_make', 'vehicle_year', 'vehicle_ownership']


    # In[22]:


    df = df[predictor + target_variable].copy()




    dum_df = pd.get_dummies(df, columns=categorical_variables)



    dum_df['last_name'] = dum_df['last_name'].astype('category')


    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    dum_df['last_name_Cat'] = labelencoder.fit_transform(dum_df['last_name'])


    # In[26]:


    dum_df[['last_name','last_name_Cat']]


    # In[27]:


    dum_df.drop(['last_name', 'last_name_Cat'], axis=1)




    def calculate_training_accuracy(X_train, y_train, model):
        '''returns in-sample accuracy for already fit model.'''
        predictions = model.predict(X_train)
        error = accuracy_score(y_train, predictions)
        return error

    def calculate_validation_accuracy(X_test, y_test, model):
        '''returns out-of-sample accuracy for already fit model.'''
        predictions = model.predict(X_test)
        error = accuracy_score(y_test, predictions)
        return error

    def calculate_metrics_accuracy(X_train, y_train, X_test, y_test, model):
        '''fits model and returns the F1-Score for in-sample error and out-of-sample error'''
        model.fit(X_train, y_train)
        train_error = calculate_training_accuracy(X_train, y_train, model)
        validation_error = calculate_validation_accuracy(X_test, y_test, model)
        return train_error, validation_error

    def calculate_training_error(X_train, y_train, model):
        '''returns in-sample error for already fit model.'''
        predictions = model.predict(X_train)
        error = f1_score(y_train, predictions,  average='weighted')
        return error

    def calculate_validation_error(X_test, y_test, model):
        '''returns out-of-sample error for already fit model.'''
        predictions = model.predict(X_test)
        error = f1_score(y_test, predictions,  average='weighted')
        return error

    def calculate_metrics(X_train, y_train, X_test, y_test, model):
        '''fits model and returns the F1-Score for in-sample error and out-of-sample error'''
        model.fit(X_train, y_train)
        train_error = calculate_training_error(X_train, y_train, model)
        validation_error = calculate_validation_error(X_test, y_test, model)
        return train_error, validation_error


    # Splitting the data into training and test dataset.

    # In[29]:


    data, X_test, target, y_test = train_test_split(dum_df.drop(['last_name', 'last_name_Cat'], axis=1),dum_df['last_name_Cat'], shuffle=True, test_size=0.3, random_state=15)


    # We are gonna use 10-Fold cross validation to tune our model.

    # In[30]:

    model = RandomForestClassifier(n_estimators= 200, n_jobs=-2)
    model.fit(data, target)
    joblib.dump(model, "model.joblib")


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load("model.joblib")
    return model
