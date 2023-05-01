from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, confusion_matrix
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random


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

# create LabelEncoder object
le = LabelEncoder()

# Splitting into input and output data
X = df.drop(['NOT_HOSPITALIZED', 'ICU', 'DIED'], axis=1)
y = pd.get_dummies(df[['NOT_HOSPITALIZED', 'ICU', 'DIED']])

sample_df = y.sample(n=50)
print(sample_df)



# -- Neural Network --

# Creating the model
model = Sequential()
model.add(Dense(14, input_dim=16, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
epochs = 1
batch_size = 256
model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=0)

model.save('COVID_Predictor_May1-23.h5')

# self test

df_test = df.sample(n=250000)
X_test = df_test.drop(['NOT_HOSPITALIZED', 'ICU', 'DIED'], axis=1)
y_test = pd.get_dummies(df_test[['NOT_HOSPITALIZED', 'ICU', 'DIED']])

score = model.evaluate(X_test ,y_test , verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Make predictions on the test set
y_pred = model.predict(X_test, verbose=0)
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


"""
results = []

for epoch in range(1,11):
    print(f'now testing with epoch size {epoch}')
    model.fit(X, y, epochs=epoch, batch_size=1024, validation_split=0.2, verbose=1)
    acc_scores = []
    loss_scores = []
    for run in range(5):
        print(f'test run {run} for epoch size {epoch}')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        score = model.evaluate(X_test ,y_test , verbose=0)
        acc_scores.append(score[1])
        loss_scores.append(score[0])
    results.append((sum(acc_scores)/len(acc_scores), sum(loss_scores)/len(loss_scores)))

print(results)
"""
