import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from main import *
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=14784*1, out_features=2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        # self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

# Load the data using DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print('GPU is available.')
else:
    device = torch.device("cpu")
    print('GPU is not available, running on CPU.')

meta_data = read_csv('data/metabolism.csv')
spect_data = read_csv('data/spectralIndex.csv')
meta_data = np.array(meta_data)
spect_data = np.array(spect_data)

meta_name = meta_data[0, 1:]
meta_data = meta_data[1:, 1:]
spect_data = spect_data[1:, 1:]

meta_data = meta_data.astype('float32')
spect_data = spect_data.astype('float32')


model = ShallowCNN()

model.to(device)

good_number = 0

for i in tqdm(range(528)):
    predict_meta = [x[21] for x in meta_data]
    X_train, X_test, y_train, y_test = train_test_split(spect_data, predict_meta, test_size=0.2, random_state=2)


    X_train = np.reshape(X_train, (len(X_train), 1, 1848))
    y_train = np.reshape(y_train, (len(y_train), 1))



    data = torch.tensor(X_train)
    data = data.to(device)
    data = data.cuda()
    # print(data[0])
    labels = torch.tensor(y_train)
    labels = labels.to(device)
    labels = labels.cuda()



    dataset = TensorDataset(data, labels)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize the model

    # Define the loss function and optimizer
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Train the model
    r2 = []
    X_test = torch.tensor(X_test)
    X_test = np.reshape(X_test, (len(X_test), 1, 1848))

    for epoch in range(20):
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                X_test = X_test.to(device)
                outputs2 = model(X_test)
                outputs2 = torch.tensor(outputs2, dtype=torch.float32)
                outputs2 = outputs2.cpu().numpy()
                r2.append(r2_score(y_test, outputs2))
        print(max(r2))

    if max(r2) >= 0.3:
        good_number += 1

print(good_number)
