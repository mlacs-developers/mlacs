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
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    msg = "You need PyTorch installed to use Neural Networks model"
    raise ImportError(msg)


default_parameters = {"hiddenlayers": [],
                      "activation": "linear",
                      "epoch": 5000,
                      "optimizer": "l-bfgs",
                      "learning_rate": 1e-2,
                      "batch_size": 5,
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
        # TODO scale with respect to elements
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

        ecoef = self.energy_coefficient / np.std(ymat_e)
        fcoef = self.forces_coefficient / np.std(ymat_f)

        msg = "number of atomic environment for training:  " + \
              f"{amat_e.shape[0]}\n"

        optimizer = _get_optimizers(self.parameters["optimizer"],
                                    self.neuralnetwork.parameters(),
                                    self.parameters["learning_rate"])
        loss_fn = _get_loss_fn(self.parameters["loss_function"])

        dataset = Data(amat_e, amat_f, ymat_e, ymat_f, idx_e, idx_df)
        loader = DataLoader(dataset,
                            batch_size=self.parameters["batch_size"],
                            shuffle=True,
                            collate_fn=_collat_fn)

        iepoch = []
        iloss = []
        for epoch in range(self.parameters["epoch"]):
            for data in loader:
                x_e = data[0].requires_grad_()
                x_f = data[1]
                y_e = data[2]
                y_f = data[3]
                idx_e = data[4]
                idx_f = data[5]
                nat = data[6]
                if self.parameters["optimizer"] == "l-bfgs":
                    def closure():
                        pred_e, pred_f = self.neuralnetwork(x_e, x_f,
                                                            idx_f, idx_e)
                        loss_e = ecoef * loss_fn(pred_e / nat,
                                                 y_e / nat)
                        loss_f = fcoef * loss_fn(pred_f.flatten(),
                                                 y_f.flatten())
                        loss = loss_e + loss_f
                        optimizer.zero_grad()
                        loss.backward()
                        return loss
                    loss = optimizer.step(closure).data
                else:
                    pred_e, pred_f = self.neuralnetwork(x_e, x_f,
                                                        idx_f, idx_e)
                    loss_e = ecoef * loss_fn(pred_e / nat,
                                             y_e / nat)
                    loss_f = fcoef * loss_fn(pred_f.flatten(),
                                             y_f.flatten())
                    loss = loss_e + loss_f
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            iepoch.append(epoch)
            iloss.append(loss.item())

        # We need to reorganize results so that parameter are printed right :
        # lay0.node0.bias then lay0.node0.weight then layer0.nod1.bias then ...
        results = {}
        for iel, el in enumerate(self.elements):
            nparams = 0
            results_el = {}
            ilay = 0
            results_el["scale0"] = scale0  # TODO scale per element
            results_el["scale1"] = scale1  # TODO scale per element
            for i, lay in enumerate(self.neuralnetwork.network[iel].layers):
                if hasattr(lay, "weight"):  # The act layers don't have weight
                    bias = lay.bias.detach().numpy()
                    weight = lay.weight.detach().numpy()
                    param = np.c_[bias[:, None], weight]
                    results_el[f"layer{ilay}"] = param.flatten()
                    ilay += 1
                    nparams += param.size
            results[el] = results_el

        loader = DataLoader(dataset,
                            batch_size=len(dataset) + 1,
                            shuffle=False,
                            collate_fn=_collat_fn)

        self.write_mlip(results, nparams,
                        self.neuralnetwork.network[0].nnodes[1:],
                        self.neuralnetwork.network[0].func)
        msg = self.compute_tests(loader, msg)

        epochloss = np.c_[iepoch, iloss]
        header = "epoch    Total loss"
        np.savetxt("MLIP-Loss.dat",
                   epochloss,
                   header=header, fmt="%6d  %15.10f")

        self.init_calc()
        return msg

# ========================================================================== #
    def compute_tests(self, loader, msg):
        """
        """
        for data in loader:
            x_e = data[0].requires_grad_()
            x_f = data[1]
            y_e = data[2]
            y_f = data[3]
            idx_e = data[4]
            idx_f = data[5]
            nat = data[6]
            (pred_e, pred_f) = self.neuralnetwork(x_e, x_f, idx_f, idx_e)

        e_true = y_e.detach().numpy() / nat.detach().numpy()
        e_mlip = pred_e.detach().numpy() / nat.detach().numpy()
        f_true = y_f.detach().numpy()
        f_mlip = pred_f.detach().numpy()

        # Compute RMSE and MAE
        rmse_energy = np.sqrt(np.mean((e_true - e_mlip)**2))
        mae_energy = np.mean(np.abs(e_true - e_mlip))

        rmse_forces = np.sqrt(np.mean((f_true - f_mlip)**2))
        mae_forces = np.mean(np.abs(f_true - f_mlip))

        # Prepare message to the log
        msg += f"RMSE Energy    {rmse_energy:.4f} eV/at\n"
        msg += f"MAE Energy     {mae_energy:.4f} eV/at\n"
        msg += f"RMSE Forces    {rmse_forces:.4f} eV/angs\n"
        msg += f"MAE Forces     {mae_forces:.4f} eV/angs\n"
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
        # idx_df -> array with iconf, iat, jat, iel
        idx_df = np.array(descriptor[2], dtype=int)[::3]
        idx_df = np.c_[np.ones(idx_df.shape[0], dtype=int) * self.nconfs,
                       idx_df]

        # idx_e -> array with iconf, iat, iel
        idx_e = np.array(descriptor[3], dtype=int)
        idx_e = np.c_[np.ones(natoms, dtype=int) * self.nconfs,
                      idx_e]

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
                                           self.perat_desc,
                                           len(self.elements))


# ========================================================================== #
class NeuralNetwork(nn.Module):
    """
    """
    def __init__(self, hiddenlayers, activation,
                 ndescriptor, nelements):
        super(NeuralNetwork, self).__init__()
        self.activation = []
        self.nelements = nelements
        self.ndescriptor = ndescriptor

        self.network = nn.ModuleList()
        for i in range(nelements):
            self.network.append(NeuralNetworkPerElement(hiddenlayers,
                                                        activation,
                                                        ndescriptor))

# ========================================================================== #
    def forward(self,  x_e, x_f, idx_df, idx_e):
        """
        """
        confs = np.unique(idx_e[:, 0])
        nconfs = confs.shape[0]

        pred_e = torch.zeros((nconfs))
        pred_f = torch.zeros((x_e.shape[0], 3))
        for iel in range(self.nelements):
            el_pred = self.network[iel](x_e, x_f,
                                        idx_df, idx_e, iel)
            pred_e = pred_e + el_pred[0]
            pred_f = pred_f + el_pred[1]
        return pred_e, pred_f


# ========================================================================== #
# ========================================================================== #
class NeuralNetworkPerElement(nn.Module):
    """
    """
    def __init__(self, hiddenlayers, activation, ndescriptor):
        super(NeuralNetworkPerElement, self).__init__()
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
    def forward(self, x_e, x_f, idx_df, idx_e, iel):
        """
        Does the forward stuff and return the energy and the forces
        Since the forces need the derivative of the output wrt the descriptor,
        it gets a bit messy.
        To gain speed, everything is organized so the number of loop is minimal
        -> This means that everything is index-related -> a bit hard to read
        Sorry
        """
        confs = np.unique(idx_e[:, 0])
        nconfs = confs.shape[0]

        # We need a mask to compute stuff only for current element
        mask_e = idx_e[:, 1] == iel

        # We prepare the tensors for the results
        pred_e = torch.zeros((nconfs))
        pred_f = torch.zeros((x_e.shape[0], 3))

        # We start by computing the energy, with the element mask
        pred_e.index_add_(0, idx_e[:, 0].to(torch.int64),
                          self.layers(x_e).squeeze() * mask_e)

        # Now we compute dF(x)/x -> create array of size (natom, ndesc)
        dedx = grad(self.layers(x_e), x_e,
                    grad_outputs=torch.ones_like(self.layers(x_e)),
                    create_graph=True)[0]
        dedx = dedx * mask_e[:, None]  # We mask to have only der of f(x(R_j))

        dedx_neigh = dedx[idx_df[:, 0].to(torch.int64)]
        dedx_neigh = dedx_neigh.unsqueeze(1).repeat(1, 3, 1)  # shape dxdr

        # And boom, we have sum_k dF(x(R)_k)/dx(R)_k * dX(R)_k/dR_j
        f_contrib = torch.mul(dedx_neigh, x_f).sum(dim=2)

        # And voila, the predicted forces in shape (natoms, 3)
        pred_f.index_add_(0, idx_df[:, 1].to(torch.int64), f_contrib)
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
class Data(Dataset):
    """
    """
    def __init__(self, amat_e, amat_f, ymat_e, ymat_f, idx_e, idx_df):
        self.amat_e = torch.from_numpy(amat_e.astype(np.float32))
        self.amat_f = torch.from_numpy(amat_f.astype(np.float32))
        self.ymat_e = torch.from_numpy(ymat_e.astype(np.float32))
        self.ymat_f = torch.from_numpy(ymat_f.astype(np.float32))
        self.idx_df = torch.from_numpy(idx_df.astype(np.int64))
        self.idx_e = torch.from_numpy(idx_e.astype(np.int64))

        self._length = np.max(idx_e[:, 0])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        conf_e = self.idx_e[:, 0] == idx
        conf_df = self.idx_df[:, 0] == idx

        amat_e = self.amat_e[conf_e]
        amat_f = self.amat_f[conf_df]
        ymat_e = self.ymat_e[idx]
        ymat_f = self.ymat_f[conf_e]
        idx_e = self.idx_e[conf_e]
        idx_df = self.idx_df[conf_df]
        return amat_e, amat_f, ymat_e, ymat_f, idx_e, idx_df


# ========================================================================== #
# ========================================================================== #
def _collat_fn(batch):
    """
    """
    amat_e = torch.cat([data[0] for data in batch], dim=0)
    amat_f = torch.cat([data[1] for data in batch], dim=0)
    ymat_e = torch.cat([torch.Tensor([data[2]]) for data in batch], dim=0)
    ymat_f = torch.cat([data[3] for data in batch], dim=0)
    idx_e = torch.cat([data[4] for data in batch], dim=0)
    idx_df = torch.cat([data[5] for data in batch], dim=0)

    confs = torch.unique(idx_e[:, 0])
    idx_conf_e = torch.Tensor([])
    idx_conf_dfi = torch.Tensor([])
    idx_conf_dfj = torch.Tensor([])
    natoms = torch.Tensor([])
    ii = 0
    jj = 0
    for iconf in confs:
        bool_conf_df = idx_df[:, 0] == iconf
        nat = torch.unique(idx_df[bool_conf_df, 1]).shape[0]

        tmp_idx_e = torch.ones(nat, dtype=torch.int) * int(ii)
        tmp_idx_dfi = idx_df[bool_conf_df][:, 1] + int(jj)
        tmp_idx_dfj = idx_df[bool_conf_df][:, 2] + int(jj)

        idx_conf_e = torch.cat((idx_conf_e, tmp_idx_e))
        idx_conf_dfi = torch.cat((idx_conf_dfi, tmp_idx_dfi))
        idx_conf_dfj = torch.cat((idx_conf_dfj, tmp_idx_dfj))

        natoms = torch.cat((natoms, torch.Tensor([nat])))
        ii += 1
        jj += nat

    new_idx_e = torch.column_stack((idx_conf_e, idx_e[:, 2]))
    new_idx_e = new_idx_e.to(torch.int64)
    new_idx_df = torch.column_stack((idx_conf_dfi, idx_conf_dfj))
    new_idx_df = new_idx_df.to(torch.int64)

    return amat_e, amat_f, ymat_e, ymat_f, new_idx_e, new_idx_df, natoms


# ========================================================================== #
def _get_optimizers(method, param, learning_rate):
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
    elif method == "adadelta":
        optimizer = optim.Adadelta(param,
                                   lr=learning_rate)
    else:
        msg = "The selected optimizer is not implemented"
        raise NotImplementedError(msg)
    return optimizer


# ========================================================================== #
# ========================================================================== #
def _get_loss_fn(method):
    """
    """
    if method == "mse":
        loss_fn = nn.MSELoss()
    if method == "huber":
        loss_fn = nn.HuberLoss()
    if method == "l1":
        loss_fn = nn.L1Loss()
    return loss_fn
