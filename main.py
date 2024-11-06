import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from simple_fit.simple_fit import SimpleNN
from simple_fit.simple_hdr import SimpleHdr
from util.functions import esin, TwoSin, MultiSin, torch_esin3d, torch_esin3d_loss, torch_sumAsin_loss
from util.generate import generate_hf_2d_data, generate

if __name__ == '__main__':
    # start = 1 / (4 * np.pi)
    # end = start + 2 * (1 / (2 * np.pi))
    start = 0.001
    end = 2
    x_train, y_train = generate(TwoSin(2 * np.pi, 4 * np.pi), 1000, start, end)  # generate_hf_2d_data(1000, 0.2)
    plt.figure(figsize=(10, 6))
    y2_train = MultiSin([1,1,1,1,1], [2*np.pi,4*np.pi,6*np.pi,8*np.pi,10*np.pi ])(x_train)
    plt.plot(
        x_train, y_train,
        label='Generated Data with Noise', color='blue', alpha=0.5)
    plt.plot(
        x_train, y2_train,
        label='Multisin Data with Noise', color='blue', alpha=0.5)
    plt.plot(
        x_train, np.sin(x_train),
        label='True Curve', color='red', linestyle='--')

    plt.legend()
    plt.title('Generated High Frequency Data with Noise')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.show()
    def opt(params):
        return optim.LBFGS(params, max_iter=100, tolerance_grad=1e-10, tolerance_change=1e-10,line_search_fn="strong_wolfe")
    model = SimpleNN(optim.Adam, nn.MSELoss())
    # model = SimpleHdr(opt, torch_sumAsin_loss)
    # model = SimpleHdr(optim.Adam, torch_sumAsin_loss)
    model.fit(10000, x_train, y2_train)
    y_pred = model.predict(x_train)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y2_train, label='Original Curve', color='blue')
    plt.plot(x_train, y_pred, label='Learned Curve', color='red')
    # inputs = torch.from_numpy(x_train).requires_grad_()
    # targets = torch.from_numpy(y_train)
    #
    # num_epochs = 10000
    # for epoch in range(num_epochs):
    #     # Training
    #     # loss = step.train()
    #     loss = train(model, model.optimizer(model.parameters()), model.criterion, inputs, targets)
    #
    #     if (epoch + 1) % 100 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
    plt.legend()
    plt.show()