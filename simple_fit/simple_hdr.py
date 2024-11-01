import torch
from torch import nn

from util.functions import torch_esin3d, torch_sumAsin


class SimpleHdr(nn.Module):
    def __init__(self, optimizer, criterion):
        super(SimpleHdr, self).__init__()

        self.criterion = criterion

        self.fc1 = nn.Linear(1, 10)  # 1 input feature, 10 hidden units
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(10, 10)  # 10 hidden units, 1 output
        self.fc3 = nn.Linear(10, 10)  # 10 hidden units, 1 output
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 1000)
        self.optimizer = optimizer(self.parameters())
        # self.init_xavier()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        return x

    def fit(self, num_epochs, x_train, y_train):
        inputs = torch.from_numpy(x_train).requires_grad_().float().view(-1, 1)
        targets = torch.from_numpy(y_train).float().view(-1, 1)
        step = ModelTrainingStep(self, inputs, targets)
        print(inputs)
        print(targets)
        for epoch in range(num_epochs):
            loss = step.train()
            # self.optimizer.step(closure)
            # loss = closure().item()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch_sumAsin(self(torch.from_numpy(x).float().view(-1,1))).numpy()

    def init_xavier(self):
        # torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

class ModelTrainingStep:
    def __init__(self, model, inputs, targets):
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