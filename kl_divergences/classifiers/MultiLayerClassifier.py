
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class MultiLayerClassifier(nn.Module):

    def __init__(self, input, hidden_layers=None, drop_out=0.05):
        super(MultiLayerClassifier, self).__init__()

        if hidden_layers is None:
            hidden_layers = []

        hidden_layers = [input] + hidden_layers + [1]

        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.layers += [
                nn.Linear(
                    hidden_layers[i],
                    hidden_layers[i + 1]
                )
            ]
        self.dropout = nn.Dropout(drop_out)
        # self.tanh =
        self.activation = F.relu
        # self.activation = F.relu
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.sig(self.layers[-1](x))

    def fit(self, X, y, X_test=None, y_test=None, training_stages=None):

        if training_stages is None:
            training_stages = [(0.0001, 250)]

        if not torch.is_tensor(X):
            _X = torch.tensor(X).float()
        else:
            _X = X

        if not torch.is_tensor(y):
            _y = torch.tensor(y).float()
        else:
            _y = y

        if X_test is not None and not torch.is_tensor(X_test):
            _X_test = torch.tensor(X_test).float()
        else:
            _X_test = X_test

        if y_test is not None and not torch.is_tensor(y_test):
            _y_test = torch.tensor(y_test).float()
        else:
            _y_test = y_test

        criterion = nn.BCELoss()

        train_error = []
        test_error = []

        for stage in training_stages:
            optimizer = torch.optim.Adam(self.parameters(), lr=stage[0])

            for e in range(1 + stage[1]):
                self.train()
                self.zero_grad()

                output = self.forward(_X).view(-1)
                error = criterion(output, _y)
                error.backward()

                train_error += [error.item()]

                self.eval()
                if _X_test is not None:
                    predictions = self.forward(_X_test).view(-1)

                    e2 = criterion(predictions, _y_test)
                    test_error += [e2.item()]
                optimizer.step()

        if X_test is not None:
            plt.plot([i for i in range(len(train_error))], train_error)
            plt.plot([i for i in range(len(test_error))], test_error)
            plt.legend(['train', 'test'])
            plt.show()

    def predict(self, x):
        self.eval()
        if not torch.is_tensor(x):
            _x = torch.tensor(x).float()
        else:
            _x = x

        return torch.round(self(_x)).detach().numpy().reshape(-1)


