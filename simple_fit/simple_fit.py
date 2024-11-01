import torch
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, optimizer, criterion):
        super(SimpleNN, self).__init__()
        self.criterion = criterion
        self.fc1 = nn.Linear(1, 10)  # 1 input feature, 10 hidden units
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(10, 10)  # 10 hidden units, 1 output
        self.fc3 = nn.Linear(10, 1)
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

    def fit(self, num_epochs, x_train, y_train):
        inputs = torch.from_numpy(x_train).requires_grad_().float().view(-1, 1)
        targets = torch.from_numpy(y_train).float().view(-1, 1)
        step = ModelTrainingStep(self, inputs, targets)
        print(inputs)
        print(targets)
        for epoch in range(num_epochs):
            # def closure():
            #     self.optimizer.zero_grad()
            #     outputs = self(inputs)
            #     loss = self.criterion(outputs, targets)
            #     loss.backward()
            #     return loss
            # Training
            loss = step.train()
            # self.optimizer.step(closure)
            # loss = closure().item()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(torch.from_numpy(x).float().view(-1,1)).numpy()


class ModelTrainingStep:
    def __init__(self, model: SimpleNN, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = targets

    def __call__(self):
        self.model.optimizer.zero_grad()
        outputs = self.model(self.inputs)
        loss = self.model.criterion(outputs, self.targets)
        loss.backward()
        return loss

    def train(self):
        self.model.optimizer.step(self)
        return self().item()