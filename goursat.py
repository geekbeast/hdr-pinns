import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from networks.goursat import Goursat
from simple_fit.simple_fit import SimpleNN
from simple_fit.simple_hdr import SimpleHdr
from util.functions import esin, TwoSin, MultiSin, torch_esin3d, torch_esin3d_loss, torch_sumAsin_loss
from util.generate import generate_hf_2d_data, generate

if __name__ == '__main__':
    n_int = 1024
    n_sb = 128
    n_tb = 128
    goursat = Goursat(n_int, n_sb, n_tb)
    input_sb_, output_sb_ = goursat.add_spatial_boundary_points()
    # input_tb_, output_tb_ = goursat.add_temporal_boundary_points()
    input_int_, output_int_ = goursat.add_interior_points()

    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 0].detach().numpy(), label="Boundary Points")
    plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label="Interior Points")
    # plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label="Initial Points")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    plt.show()
    n_epochs = 1
    optimizer_LBFGS = optim.LBFGS(goursat.approximate_solution.parameters(),
                                  lr=float(0.5),
                                  max_iter=500000,
                                  max_eval=500000,
                                  history_size=150,
                                  line_search_fn="strong_wolfe",
                                  tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(goursat.approximate_solution.parameters(),
                                lr=float(0.001))
    hist = goursat.fit(num_epochs=n_epochs,
                    optimizer=optimizer_LBFGS,
                    verbose=True)

    plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
    plt.xscale("log")
    plt.legend()
    plt.show()
    goursat.plotting()
