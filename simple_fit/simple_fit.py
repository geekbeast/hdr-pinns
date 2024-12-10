import torch
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, optimizer, criterion):
        super(SimpleNN, self).__init__()
        self.criterion = criterion
        self.fc1 = nn.Linear(1, 10)  # 1 input feature, 10 hidden units
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(10, 10)  # 10 hidden units, 1 output
        self.fc3 = nn.Linear(10, 10)  # 10 hidden units, 1 output
        self.fc4 = nn.Linear(10, 1)
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x

    def fit(self, num_epochs, x_train, y_train):
        inputs = torch.from_numpy(x_train).float().view(-1, 1)
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        inputs.requires_grad_()

        targets = torch.from_numpy(y_train).float().view(-1, 1)
        targets = targets.cuda() if torch.cuda.is_available() else targets

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
        x_tensor = torch.from_numpy(x).float().view(-1, 1)
        x_tensor = x_tensor.cuda() if torch.cuda.is_available() else x_tensor
        with torch.no_grad():
            return self(x_tensor).cpu().numpy()


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