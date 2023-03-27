from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


# split into X and Y

dfx = df[['FEMALE', 'AGE', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES',
          'COPD', 'ASTHMA', 'INMSUPR', 'HYPERTENSION', 'OTHER_DISEASE',
          'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO',
          'COVID_CLASS']].values

dfy = df[['NOT_HOSPITALIZED', 'ICU', 'DIED']].values

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy,test_size=0.3,random_state=0, stratify=dfy) #assign 70% of dataset to train, 30% to test



# -- Neural Network --

class C19PredNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(C19PredNetwork, self).__init__()
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Apply the first fully connected hidden layer
        x = self.fc1(x)
        x = self.activation(x)
        
        # Apply the second fully connected hidden layer
        x = self.fc2(x)
        x = self.activation(x)
        
        # Apply the output layer
        x = self.fc3(x)
        
        return x
"""
class C19PredNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(C19PredNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
"""

# Create an instance of the network
input_size = 16
hidden_size = 14
output_size = 3
model = C19PredNetwork(input_size, hidden_size, output_size)


# Hyperparameters
learning_rate = 0.01
num_epochs = 5
batch_size = 64

# Neural network dimensions
input_size = 16
hidden_size = 14
output_size = 3

model = C19PredNetwork(input_size, hidden_size, output_size)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert the training data into PyTorch tensors and create DataLoader object

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=float)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the neural network
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss/len(train_loader)))

# Convert the test data into PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=float)

# Test the performance of the trained neural network

# Compute predicted outputs for test set
y_pred = model(X_test_tensor)

# Get predicted class labels by taking the index of the max value for each sample
_, predicted = torch.max(y_pred.data, dim=1)
predicted = predicted.numpy()
y_test = y_test_tensor.numpy()

# Compute accuracy
predicted_tensor = torch.from_numpy(predicted)
y_test_tensor = torch.from_numpy(y_test)
predicted_tensor = predicted_tensor.reshape(y_test_tensor.shape)

correct = torch.eq(predicted_tensor, y_test_tensor).sum().item()
total = len(y_test)
accuracy = correct / total
print('Accuracy: {:.2%}'.format(accuracy))


"""
with torch.no_grad():
    outputs = model(X_test_tensor)
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred.data, dim=1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = 100 * correct / total
    print('Test Accuracy: %.2f %%' % accuracy)
    
"""
