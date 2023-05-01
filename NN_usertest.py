from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, confusion_matrix
import numpy as np
import pandas as pd
import random


def display_preds(predictions):
    for x in predictions:
        vals = (np.round(x, decimals=3))
        print(f'Not Hospitalized: {vals[0]}, ICU: {vals[1]}, Died: {vals[2]} ')
        

model = load_model('COVID_Predictor_May1-23.h5')
"""
dictdata = {
    'FEMALE':[0.0],
    'INTUBED':[1.0],
    'PNEUMONIA':[1.0],
    'AGE':[0.61157],
    'PREGNANT':[0.0],
    'DIABETES':[1.0],
    'COPD':[0.0],
    'ASTHMA':[0.0],
    'INMSUPR':[0.0],
    'HYPERTENSION':[1.0],
    'OTHER_DISEASE':[0.0],
    'CARDIOVASCULAR':[0.0],
    'OBESITY':[0.0],
    'RENAL_CHRONIC':[0.0],
    'TOBACCO':[0.0],
    'COVID_CLASS':[1.0]
    }

data = pd.DataFrame(dictdata)
predictions = model.predict(data,verbose=1)
print(predictions)
display_preds(predictions)

"""

df = pd.read_csv('Covid_Data.csv')
# --- Data Preprocessing ---

# drop unused columns
df = df.drop(['USMER', 'MEDICAL_UNIT',], axis=1)
print(df.columns)
print(df.head())


# clean up/clarify column names
df = df.rename(columns={'SEX': 'FEMALE'})
df = df.rename(columns={'HIPERTENSION': 'HYPERTENSION'})
df = df.rename(columns={'CLASIFFICATION_FINAL': 'COVID_CLASS'})
df = df.rename(columns={'DATE_DIED': 'DIED'})
df = df.rename(columns={'PATIENT_TYPE': 'NOT_HOSPITALIZED'})


# replace 1=yes, 2=no boolean values to usual 1 and 0 in bool columns
# replace 97,98, and 99 with 0
bool_columns = ['FEMALE', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES',
                'COPD', 'ASTHMA', 'INMSUPR', 'HYPERTENSION', 'OTHER_DISEASE',
                'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO',
                'ICU', 'NOT_HOSPITALIZED']
for category in bool_columns:
    df[category] = df[category].replace(2, 0)
    df[category] = df[category].replace(97, 0)
    df[category] = df[category].replace(98, 0)
    df[category] = df[category].replace(99, 0)


# replace DIED values with boolean vals
df.loc[df['DIED'] != '9999-99-99', 'DIED'] = 1
df['DIED'] = df['DIED'].replace('9999-99-99', 0)


# replace COVID_CLASS values above 3 with 0 (mean patient is not a carrier)
df.loc[df['COVID_CLASS'] > 3, 'COVID_CLASS'] = 0

# normalize age and covid class
scaler = MinMaxScaler()
df['AGE'] = scaler.fit_transform(df[['AGE']])
df['COVID_CLASS'] = scaler.fit_transform(df[['COVID_CLASS']])


# Splitting into input and output data
X = df.drop(['NOT_HOSPITALIZED', 'ICU', 'DIED'], axis=1)
y = pd.get_dummies(df[['NOT_HOSPITALIZED', 'ICU', 'DIED']])


df_test = df.sample(n=50000)
X_test = df_test.drop(['NOT_HOSPITALIZED', 'ICU', 'DIED'], axis=1)
y_test = pd.get_dummies(df_test[['NOT_HOSPITALIZED', 'ICU', 'DIED']])

score = model.evaluate(X_test ,y_test , verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# Make predictions on the test set
y_pred = model.predict(X_test,verbose=0)

# Convert the predicted probabilities to binary labels using a threshold of 0.5
y_pred = (y_pred > 0.5).astype(int)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"F1 score: {f1:.2f}")

# compute AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred)
print(f'AUC_ROC: {auc_roc}')

# compute MSE
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# compute confusion matrix
y_pred_binary = (y_pred > 0.5).astype(int) # convert probabilities to binary predictions
#cm = confusion_matrix(y_test, y_pred_binary)
#print(f'CM: {cm}')
