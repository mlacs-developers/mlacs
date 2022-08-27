'''
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
'''
import numpy as np

from mlacs.mlip import MlipManager
try:
    import torch
    from torch import nn
    from torch import optim
    from torch.autograd import grad
except ImportError:
    msg = "You need PyTorch installed to use Neural Networks model"
    raise ImportError(msg)


default_parameters = {"hiddenlayers": [],
                      "activation": "linear",
                      "alpha": 1e-8,
                      "epoch": 5000,
                      "optimizer": "l-bfgs",
                      "learning_rate": 1e-2,
                      "batch_size": 100,
                      "normalization": "norm",
                      "loss_function": "mse"}


# ========================================================================== #
# ========================================================================== #
class NeuralNetworkMlip(MlipManager):
    """
    Parent Class for Neural Network MLIP
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 nthrow=10,
                 parameters=None,
                 energy_coefficient=1.0,
                 stress_coefficient=1.0,
                 forces_coefficient=1.0):

        MlipManager.__init__(self,
                             atoms,
                             rcut,
                             nthrow,
                             energy_coefficient,
                             forces_coefficient,
                             stress_coefficient)

        self.parameters = default_parameters
        if parameters is not None:
            self.parameters.update(parameters)

# ========================================================================== #
    def train_mlip(self):
        """
        """
        amat_e = self.amat_e
        amat_f = self.amat_f
        ymat_e = self.ymat_e
        ymat_f = self.ymat_f
        idx_df = self.idx_df
        idx_e = self.idx_e

        # We start by scaling the data
        if self.parameters["normalization"] == "min-max":
            scale0 = amat_e.min(axis=0)
            scale1 = amat_e.max(axis=0) - scale0
        elif self.parameters["normalization"] == "norm":
            scale0 = amat_e.mean(axis=0)
            scale1 = amat_e.std(axis=0)
        elif self.parameters["normalization"] is None:
            scale0 = np.zeros(amat_e.shape[1])
            scale1 = np.ones(amat_e.shape[1])
        else:
            msg = "normalization parameters should be either " + \
                  "\"min-max\", \"norm\" or None"
            raise RuntimeError(msg)

        amat_e = (amat_e - scale0) / scale1
        amat_f = amat_f / scale1

        sigma_e = self.energy_coefficient / np.std(ymat_e) / len(ymat_e)
        sigma_f = 100000 / np.std(ymat_f) / len(ymat_f)

        msg = "number of atomic environment for training:  " + \
              f"{amat_e.shape[0]}\n"

        optimizer = get_optimizers(self.parameters["optimizer"],
                                   self.neuralnetwork.parameters(),
                                   self.parameters["learning_rate"])
        loss_fn = get_loss(self.parameters["loss_function"])

        x_e = torch.from_numpy(amat_e.astype(np.float32)).requires_grad_()
        x_f = torch.from_numpy(amat_f.astype(np.float32))
        y_e = torch.from_numpy(ymat_e.astype(np.float32))
        y_f = torch.from_numpy(ymat_f.astype(np.float32))
        idx_df = torch.from_numpy(idx_df.astype(np.int64))
        idx_e = torch.from_numpy(idx_e.astype(np.int64))

        # First let's try without batch, we'll see that later
        for epoch in range(self.parameters["epoch"]):
            if self.parameters["optimizer"] == "l-bfgs":
                def closure():
                    pred_e, pred_f = self.neuralnetwork(x_e, x_f,
                                                        idx_df, idx_e)
                    loss = sigma_e * loss_fn(pred_e, y_e)
                    loss = loss + sigma_f * loss_fn(pred_f, y_f)
                    optimizer.zero_grad()
                    loss.backward()
                    print(epoch, loss.item(),
                          pred_f[0, 0].item(), y_f[0, 0].item())
                    return loss
                optimizer.step(closure)
            else:
                (pred_e, pred_f) = self.neuralnetwork(x_e, x_f,
                                                      idx_df, idx_e)
                loss = sigma_e * loss_fn(pred_e, y_e)
                loss = loss + sigma_f * loss_fn(pred_f, y_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(epoch, loss.item())

        # We need to reorganize results so that parameter are printed right :
        # lay0.node0.bias then lay0.node0.weight then layer0.nod1.bias then ...
        results = {}
        ilay = 0
        for i, lay in enumerate(self.neuralnetwork.layers):
            if hasattr(lay, "weight"):  # The act layers don't have weight
                bias = lay.bias.detach().numpy()
                weight = lay.weight.detach().numpy()
                param = np.c_[bias[:, None], weight]
                results[f"layer{ilay}"] = param.flatten()
                ilay += 1

        amat_e = self.amat_e
        amat_f = self.amat_f
        ymat_e = self.ymat_e
        ymat_f = self.ymat_f
        amat_e = (amat_e - scale0) / scale1
        amat_f = amat_f / scale1

        x_e = torch.from_numpy(amat_e.astype(np.float32)).requires_grad_()
        x_f = torch.from_numpy(amat_f.astype(np.float32))
        y_e = torch.from_numpy(ymat_e.astype(np.float32))
        y_f = torch.from_numpy(ymat_f.astype(np.float32))

        msg = self.compute_tests(x_e, x_f, y_e, y_f,
                                 idx_df, idx_e, msg)
        self.write_mlip(scale0, scale1, results,
                        self.neuralnetwork.nnodes[1:],
                        self.neuralnetwork.func)
        self.init_calc()
        return msg

# ========================================================================== #
    def compute_tests(self, amat_e, amat_f, ymat_e, ymat_f,
                      idx_df, idx_e,  msg):
        """
        """
        (pred_e, pred_f) = self.neuralnetwork(amat_e, amat_f, idx_df, idx_e)

        confs = np.unique(idx_e[:, 0])
        nconfs = confs.shape[0]
        nat = []
        for iconf in range(nconfs):
            idx_confe = idx_e[:, 0] == iconf
            nat.append(len(np.unique(idx_e[idx_confe, 1])))
        nat = np.array(nat)
        e_true = ymat_e.detach().numpy() / nat
        e_mlip = pred_e.detach().numpy() / nat
        f_true = ymat_f.detach().numpy()
        f_mlip = pred_f.detach().numpy()

        # Compute RMSE and MAE
        rmse_energy = np.sqrt(np.mean((e_true - e_mlip)**2))
        mae_energy = np.mean(np.abs(e_true - e_mlip))

        rmse_forces = np.sqrt(np.mean((f_true - f_mlip)**2))
        mae_forces = np.mean(np.abs(f_true - f_mlip))

        # Prepare message to the log
        msg += "RMSE Energy    {:.4f} eV/at\n".format(rmse_energy)
        msg += "MAE Energy     {:.4f} eV/at\n".format(mae_energy)
        msg += "RMSE Forces    {:.4f} eV/angs\n".format(rmse_forces)
        msg += "MAE Forces     {:.4f} eV/angs\n".format(mae_forces)
        header = f"rmse: {rmse_energy:.5f} eV/at,    " + \
                 f"mae: {mae_energy:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat",
                   np.vstack((e_true, e_mlip)).T,
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_forces:.5f} eV/angs   " + \
                 f"mae: {mae_forces:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat",
                   np.vstack((f_true.flatten(), f_mlip.flatten())).T,
                   header=header, fmt="%25.20f  %25.20f")
        return msg

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        natoms = len(atoms)
        descriptor, data = self.compute_fit_matrix(atoms)

        # The actual data needed for the fit
        amat_e = descriptor[0]
        amat_f = descriptor[1].reshape(-1, 3, descriptor[1].shape[1])
        ymat_e = np.atleast_1d(data[0])
        ymat_f = data[1].reshape(-1, 3)

        # Now we need the arrays to get everything right
        # idx_df -> array with iconf, iat, jat
        idx_df = np.array(descriptor[2], dtype=int)[::3]
        idx_df = np.c_[np.ones(idx_df.shape[0], dtype=int) * self.nconfs,
                       idx_df]

        # idx_e -> array with iconf, iat
        idx_e = np.zeros((natoms, 2), dtype=int)
        idx_e[:, 0] = self.nconfs
        idx_e[:, 1] = np.arange(natoms)

        if self.nconfs == 0:
            self.amat_e = amat_e
            self.amat_f = amat_f
            self.ymat_e = ymat_e
            self.ymat_f = ymat_f
            self.idx_df = idx_df
            self.idx_e = idx_e
            self.natoms = np.array([natoms])
        else:
            self.amat_e = np.r_[self.amat_e, amat_e]
            self.amat_f = np.r_[self.amat_f, amat_f]
            self.ymat_e = np.r_[self.ymat_e, ymat_e]
            self.ymat_f = np.r_[self.ymat_f, ymat_f]
            self.idx_df = np.r_[self.idx_df, idx_df]
            self.idx_e = np.r_[self.idx_e, idx_e]
            self.natoms = np.append(self.natoms, natoms)
        self.nconfs += 1

# ========================================================================== #
    def _initialize_nn(self):
        """
        """
        self.neuralnetwork = NeuralNetwork(self.parameters["hiddenlayers"],
                                           self.parameters["activation"],
                                           self._get_ncolumns())


# ========================================================================== #
class NeuralNetwork(nn.Module):
    """
    """
    def __init__(self, hiddenlayers, activation, ndescriptor):
        super(NeuralNetwork, self).__init__()
        self.activation = []

        self.nnodes = [ndescriptor]
        self.nnodes.extend(hiddenlayers)
        self.nnodes.append(1)
        self.nlayer = len(self.nnodes) - 1

        if isinstance(activation, str):
            self.func = [activation] * (self.nlayer - 1) + ["linear"]
        else:
            self.func = activation + ["linear"]

        layers = []
        for i in range(self.nlayer):
            layers.append(nn.Linear(self.nnodes[i],
                                    self.nnodes[i+1]))
            if self.func[i] != "linear":
                layers.append(self._get_activation(self.func[i]))
        self.layers = nn.Sequential(*layers)

        for lay in layers:
            if isinstance(lay, nn.Linear):
                nn.init.xavier_uniform_(lay.weight)

# ========================================================================== #
    def forward(self, x_e, x_f, idx_df, idx_e):
        """
        Does the forward stuff and return the energy and the forces
        Since the forces need the derivative of the output wrt the descriptor,
        it gets a bit messy.
        To gain speed, everything is organized so the number of loop is minimal
        -> This means that everything is index-related -> a bit hard to read
        Sorry
        """
        confs = np.unique(idx_df[:, 0])
        nconfs = confs.shape[0]

        # We prepare the tensors for the results
        pred_e = torch.zeros((nconfs))
        pred_f = torch.zeros((x_e.shape[0], 3))

        # We start by computing the energy
        pred_e.index_add_(0, idx_e[:, 0], self.layers(x_e).squeeze())  # tadaaa

        # Now we compute dF(x)/x -> create array of size (natom, ndesc)
        dedx = grad(self.layers(x_e), x_e,
                    grad_outputs=torch.ones_like(self.layers(x_e)),
                    create_graph=True)[0]

        # Now we need to organize everything in shape
        # -> create indices from dxdr to iat_iconf
        # TODO have all indices before in order to remove the loop
        # This can be done with the batch part
        idx = torch.Tensor([])
        idx_neigh = torch.Tensor([])
        ii = 0
        for iconf in confs:
            idx_conf = idx_df[:, 0] == iconf
            nat = len(np.unique(idx_df[idx_conf, 1]))
            tmp_idx = idx_df[idx_conf][:, 2] + ii
            idx = torch.concat((idx, tmp_idx))
            tmp_idx_neigh = idx_df[idx_conf][:, 1] + ii
            idx_neigh = torch.concat((idx_neigh, tmp_idx_neigh))
            ii += nat

        # -> we get dF(x(R_i))/dx
        dedx_neigh = dedx[idx_neigh.to(torch.int64)]
        dedx_neigh = dedx_neigh.unsqueeze(1).repeat(1, 3, 1)  # shape dxdr

        # And boom, we have sum_k dF(x(R)_k)/dx(R)_k * dX(R)_k/dR_j
        # -> but shaped as dxdr
        f_contrib = torch.mul(dedx_neigh, x_f).sum(dim=2)

        # And voila, the predicted forces in shape (natoms, 3)
        pred_f.index_add_(0, idx.to(torch.int64), f_contrib)
        return pred_e, pred_f

# ========================================================================== #
    def _get_activation(self, func):
        """
        """
        if func == "tanh":
            activation_func = nn.Tanh()
        elif func == "linear":
            activation_func = None
        elif func == "relu":
            activation_func = nn.ReLU()
        elif func == "sigmoid":
            activation_func = nn.Sigmoid()
        else:
            msg = "Only linear, tanh, relu or sigmoid " + \
                  "activation functions are available"
            raise NotImplementedError(msg)

        return activation_func


# ========================================================================== #
# ========================================================================== #
def get_optimizers(method, param, learning_rate):
    """
    Function to get the optimizer for the gradient descent
    """
    if method == "l-bfgs":
        optimizer = optim.LBFGS(param,
                                lr=learning_rate)
    elif method == "sgd":
        optimizer = optim.SGD([{"params": param}],
                              lr=learning_rate)
    elif method == "adam":
        optimizer = optim.Adam(param,
                               lr=learning_rate)
    else:
        msg = "The selected optimizer is not implemented"
        raise NotImplementedError(msg)
    return optimizer


# ========================================================================== #
# ========================================================================== #
def get_loss(method):
    """
    """
    if method == "mse":
        loss_fn = nn.MSELoss()
    if method == "huber":
        loss_fn = nn.HuberLoss()
    if method == "l1":
        loss_fn = nn.L1Loss()
    return loss_fn
