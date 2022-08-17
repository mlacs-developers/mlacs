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
    from torch.utils.data import Dataset, DataLoader
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
                      "normalization": "norm"}


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

        self._initialize_parameters(parameters)

# ========================================================================== #
    def train_mlip(self):
        """
        """
        idx = self._get_idx_fit()

        """
        sigma_e = np.std(self.amatrix_energy[idx:, 1:])
        sigma_f = np.std(self.amatrix_forces[idx*3*self.natoms:, 1:])
        sigma_s = np.std(self.amatrix_stress[idx*6:, 1:])
        ecoef = self.energy_coefficient / sigma_e / \
            len(self.amatrix_energy[idx:])
        fcoef = self.forces_coefficient / sigma_f / \
            len(self.amatrix_forces[idx*3*self.natoms:])
        scoef = self.stress_coefficient / sigma_s / \
            len(self.amatrix_stress[idx*6:])
        """

        amat, ymat = self._construct_fit_matrix(idx)

        if self.parameters["normalization"] == "min-max":
            scale0 = self.amatrix_energy[idx:, 1:].min(axis=0)
            scale1 = self.amatrix_energy[idx:, 1:].max(axis=0) - scale0
        elif self.parameters["normalization"] == "norm":
            scale0 = self.amatrix_energy[idx:, 1:].mean(axis=0)
            scale1 = self.amatrix_energy[idx:, 1:].std(axis=0)
        elif self.parameters["normalization"] is None:
            scale0 = np.zeros(amat.shape[2])
            scale1 = np.ones(amat.shape[2])
        else:
            msg = "normalization parameters should be either " + \
                  "\"min-max\", \"norm\" or \"None\""
            raise RuntimeError(msg)
        amat[:, 0] = (amat[:, 0] - scale0) / scale1
        amat[:, 1:] /= scale1

        msg = "number of configurations for training:  " + \
              f"{amat.shape[0]}\n"

        # We need to prepare some stuffs before training
        data = Data(amat, ymat)
        dataloader = DataLoader(dataset=data,
                                batch_size=self.parameters["batch_size"],
                                shuffle=True)
        optimizer = self._get_optimizers()
        loss_fn = nn.MSELoss()

        # Let's train everything
        for epoch in range(self.parameters["epoch"]):
            for i, (x, y) in enumerate(dataloader):
                if self.parameters["optimizer"] == "l-bfgs":
                    def closure():
                        optimizer.zero_grad()
                        predict = self.neuralnetwork(x)
                        loss = loss_fn(predict, y)
                        print(loss.item())
                        loss.backward()
                        return loss
                    optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    predict = self.neuralnetwork(x)
                    loss = loss_fn(predict, y)
                    print(loss.item())
                    loss.backward()
                    optimizer.step()

        # We need to reorganize results so that parameter are printed right :
        # layer0.bias then layer0.weight then layer1.bias then ...
        results = {}
        ilay = 0
        for i, lay in enumerate(self.neuralnetwork.layers):
            lay = self.neuralnetwork.layers[i]
            if hasattr(lay, "weight"):
                bias = lay.bias.detach().numpy()
                weight = lay.weight.detach().numpy()
                param = np.hstack((bias[:, None], weight))
                results[f"layer{ilay}"] = param.flatten()
                ilay += 1

        msg = self.compute_tests(data.x, data.y, msg)
        self.write_mlip(scale0,
                        scale1,
                        results,
                        self.neuralnetwork.nnodes[1:],
                        self.neuralnetwork.func)
        exit()
        self.init_calc()
        return msg

# ========================================================================== #
    def _construct_fit_matrix(self, idx):
        """
        In this function, we reform matrix in shape (nconf, ndata, ndesc)
        """
        amatrix_energy = self.amatrix_energy[idx:, 1:]
        amatrix_forces = self.amatrix_forces[idx*3*self.natoms:, 1:]
        ymatrix_energy = self.ymatrix_energy[idx:]
        ymatrix_forces = self.ymatrix_forces[idx*3*self.natoms:]

        nconfs = amatrix_energy.shape[0]
        ndesc = amatrix_energy.shape[1]
        amat = np.zeros((nconfs, self.natoms*3+1, ndesc))
        ymat = np.zeros((nconfs, self.natoms*3+1))

        for iconf in range(nconfs):
            amat[iconf, 0] = amatrix_energy[iconf]
            ymat[iconf, 0] = ymatrix_energy[iconf]
            idx1 = iconf*3*self.natoms
            idx2 = (iconf+1)*3*self.natoms
            amat[iconf, 1:] = amatrix_forces[idx1:idx2]
            ymat[iconf, 1:] = ymatrix_forces[idx1:idx2]
        return amat, ymat

# ========================================================================== #
    def compute_tests(self, amat, ymat, msg):
        """
        """
        predict = self.neuralnetwork(amat)

        e_true = ymat[:, 0].detach().numpy()
        f_true = ymat[:, 1:].detach().numpy()
        e_mlip = predict[:, 0].detach().numpy()
        f_mlip = predict[:, 1:].detach().numpy()

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
                   header=header)
        header = f"rmse: {rmse_forces:.5f} eV/angs   " + \
                 f"mae: {mae_forces:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat",
                   np.vstack((f_true.flatten(), f_mlip.flatten())).T,
                   header=header)
        return msg

# ========================================================================== #
    def _get_optimizers(self):
        """
        """
        param = self.neuralnetwork.parameters()
        if self.parameters["optimizer"] == "l-bfgs":
            optimizer = optim.LBFGS(param,
                                    lr=self.parameters["learning_rate"])
        elif self.parameters["optimizer"] == "sgd":
            optimizer = optim.SGD([{"params": param}],
                                  lr=self.parameters["learning_rate"])
        elif self.parameters["optimizer"] == "adam":
            optimizer = optim.Adam(param,
                                   lr=self.parameters["learning_rate"])
        else:
            msg = "The selected optimizer is not implemented"
            raise NotImplementedError(msg)
        return optimizer

# ========================================================================== #
    def _initialize_parameters(self, parameters):
        """
        """
        self.parameters = default_parameters
        if parameters is not None:
            self.parameters.update(parameters)

# ========================================================================== #
    def _initialize_nn(self):
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

# ========================================================================== #
    def forward(self, x):
        """
        """
        desc = x[:, 0].requires_grad_()
        dxdr = x[:, 1:]

        # Predict energy and forces with current parameters
        # we need to compute the derivative of the output with respect
        # to the input to compute the forces
        predict = torch.zeros((x.shape[0], x.shape[1]))
        for iconf, x_tmp in enumerate(desc):
            e = self.layers(x_tmp).sum()
            dedx = torch.autograd.grad(e, x_tmp, create_graph=True)[0]
            predict[iconf, 0] = e
            predict[iconf, 1:] = torch.einsum("j,ij->i", dedx, dxdr[iconf])
        return predict

# ========================================================================== #
    def _get_activation(self, func):
        """
        """
        if func == "tanh":
            activation_func = nn.Tanh()
        elif func == "linear":
            activation_func = None
        elif func == "relu":
            activation_func = nn.Relu()
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
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.x.shape[0]

# ========================================================================== #
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# ========================================================================== #
    def __len__(self):
        return self.len
