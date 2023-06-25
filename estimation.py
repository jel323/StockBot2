import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import train
import dataset
from abc import ABC, abstractmethod
import loss


def getardata2(data: np.ndarray, n_before: int):
    """
    Used for formatting evaluation data, data is given as an np.ndarray, with
    shape - (nstocks, ndays, 4), with last dim being (open, high, low, close)
    """
    inp_d = np.empty((data.shape[0], data.shape[1] - n_before + 1, 4 * n_before))
    for k in range(data.shape[1] - n_before + 1):
        inp_d[:, k] = data[:, k : k + 10, :].reshape(data.shape[0], 4 * n_before)
    return inp_d


def addest(input: np.ndarray, thta: np.ndarray):
    mout = thta @ input.T
    out = np.zeros((input.shape[0], input.shape[1] + 1))
    out[:, :-1] = input
    out[:, -1] = mout
    return out


def steprls(psi, y, thta, P, rho, c):
    Ppsi = P @ psi
    nthta = thta + Ppsi / ((rho / c) + psi @ Ppsi) * (y - psi @ thta)
    nP = (1 / rho) * (P - (np.outer(Ppsi, psi @ P)) / ((rho / c) + psi @ Ppsi))
    return nthta, nP


def rlsestimate(psi, y, rho, c, plotconv=False):
    rho = 0.985
    c = 0.4
    T = psi.shape[0]
    L = psi.shape[1]
    thta = np.zeros((T + 1, L))
    thta[0] = np.random.normal(size=(L,))
    P = np.zeros((T + 1, L, L))
    P[0] = np.diag(np.abs(np.random.normal(loc=2, size=(L,))))
    for i in range(T):
        thta[i + 1], P[i + 1] = steprls(psi[i], y[i], thta[i], P[i], rho, c)
    if plotconv:
        for k in range(thta[-1].shape[0]):
            plt.figure()
            plt.plot(thta[:, k])
            plt.title(f"Convergence of RLS Parameter {k}")
            plt.xlabel("Iteration #")
            plt.ylabel("Parameter Value")
            plt.show()
    return thta[-1], P[-1]


def ls(input, output):
    return np.linalg.inv(input.T @ input) @ input.T @ output


def priceest(data, thta, n_after=5, n_before=10):
    input, output = getardata2(data, n_before)
    mout = np.zeros((n_after, output.shape[0]))
    inputl = [input]
    for k in range(n_after):
        mout[k] = thta[k] @ inputl[k].T
        inputl.append(addest(inputl[k], thta[k]))
    return mout, output


def makearmodel(
    data: np.ndarray,
    n_after=5,
    n_before=10,
    rho=0.985,
    c=0.4,
    type="rls",
    plotconv=False,
):
    input, output = dataset.gettrainingdata_np(data, n_after, n_before)

    thta = []
    P = []
    inputl = []
    inputl.append(input)
    pc = [plotconv] + [False for k in range(n_after - 1)]

    for k in range(n_after):
        if type == "rls":
            ph1, ph2 = rlsestimate(inputl[k], output[:, k], rho, c, pc[k])
            thta.append(ph1)
            P.append(ph2)
        elif type == "ls":
            thta.append(ls(inputl[k], output[:, k]))
        inputl.append(addest(inputl[k], thta[k]))
    return thta


def eval_ls(thta, data, n_after, n_before):
    input = getardata2(data, n_before)
    mout = np.zeros((data.shape[0], n_after))
    for i in range(data.shape[0]):
        inputl = [input[i]]
        for k in range(n_after):
            mout[i, k] = thta[i][k] @ inputl[k].T
            inputl.append(addest(inputl[k], thta[i][k]))
    return mout


def avgerror(real, estimate, err_func):
    errors = np.empty((real.shape[0]))
    for k in range(real.shape[0]):
        errors[k] = err_func(real[k], estimate[k])
    return errors.mean()


def mad(real, estimate):
    return np.sum(np.abs((real - estimate) / real)) / real.shape[0]


def mse(real, estimate):
    return np.sum(np.square((real - estimate) / real)) / real.shape[0]


class PriceEstimation(ABC):
    models: list
    n_after: int

    def __init__(self, data: np.ndarray, n_after: int, n_before: int):
        """
        ### Parameters

        data: np.ndarray - data for the price estimation models to
            be trained on (historical data)

        n_after: int - number of day predictions to be made (# output days)

        n_before: int - number of days predictions take in (# input days)
        """
        self.data = data
        self.nstocks = self.data.shape[0]
        self.n_after = n_after
        self.n_before = n_before
        return

    @abstractmethod
    def train(self):
        """
        This method trains the price estimator on the given data in
        initialization. These models are saved under self.models (list).
        """
        return

    @abstractmethod
    def nextdayest(self) -> np.ndarray:
        """
        Applies the trained models to predict the day following the days of
        data contained in self.data. Should use self.addnewdata before
        running this method to test on new data.
        """
        return

    def addnewdata(self, new_data: np.ndarray):
        """
        Adds a new day of data to the end of self.data.
        """
        data = np.empty(
            (
                self.data.shape[0],
                self.data.shape[1] + 1,
                5,
            )
        )
        data[:, :-1] = self.data
        data[:, -1] = new_data
        self.data = data
        return

    def test(self, test_data: np.ndarray, plot=False):
        """
        Testing function for
        """
        model_out = np.ndarray((self.nstocks, test_data.shape[1]))
        for k in range(test_data.shape[1]):
            model_out[:, k] = self.nextdayest()
            self.addnewdata(test_data[:, k])

        mean_error = avgerror(test_data[:, :, 3], model_out[:, :], mad)
        print(f"Mean Error - {mean_error}")

        if plot:
            for k in range(self.nstocks):
                plt.figure()
                plt.plot(test_data[k, :10, 3], label="Actual")
                plt.plot(model_out[k, :10], label="Estimate")
                plt.xlabel("Day")
                plt.ylabel("Price")
                plt.legend(loc="lower right")
                plt.show()
        return


class ARPriceEstimation(PriceEstimation):
    """
    Initialization parameters
    """

    def __init__(
        self,
        training_data: np.ndarray,
        n_after: int,
        n_before: int,
        type="rls",
        type_params=(0.985, 0.4),
        make_plots=False,
    ):
        super().__init__(training_data, n_after, n_before)
        self.type = type
        self.type_params = type_params
        self.make_plots = make_plots

        self.train()
        return

    def train(self):
        if self.type[-2:] == "ls":
            self.model = []
            for k in range(self.nstocks):
                self.model.append(
                    makearmodel(
                        self.data[k],
                        self.n_after,
                        self.n_before,
                        self.type_params[0],
                        self.type_params[1],
                        self.type,
                        self.make_plots,
                    )
                )
        return

    def nextdayest(self):
        model_out = np.empty((self.nstocks))
        for k in range(self.nstocks):
            model_out[k] = (
                self.model[k][0] @ self.data[k, -self.n_before :, :4].flatten().T
            )
        return model_out

    def makeestimates(self, data: np.ndarray):
        if self.type[-2:] == "ls":
            out = eval_ls(self.model, data, self.n_after, self.n_before)
        return out


class SimpleNN(nn.Module):
    def __init__(self, first_size, mid_sizes, end_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(first_size, mid_sizes[0]),
            nn.BatchNorm1d(mid_sizes[0]),
            nn.ReLU(inplace=True),
        )
        for k in range(len(mid_sizes) - 1):
            self.model.append(nn.Linear(mid_sizes[k], mid_sizes[k + 1]))
            self.model.append(nn.BatchNorm1d(mid_sizes[k + 1]))
            self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(mid_sizes[-1], end_size))
        return

    def forward(self, x):
        return 1000 * self.model(x / 1000)


class NNPriceEstimation(PriceEstimation):
    def __init__(self, data, n_after, n_before, mid_layer_sizes, make_plots=False):
        super().__init__(data, n_after, n_before)
        self.mid_sizes = mid_layer_sizes
        self.make_p = make_plots
        return

    def train(
        self,
        batch_size,
        epochs,
        learning_rate,
        learning_rate_decay,
        n_test,
        device,
        save_dir=None,
    ):
        datasets = []
        self.models = []
        for k in range(self.nstocks):
            datasets.append(
                dataset.make_both(self.data[k], self.n_after, self.n_before, n_test)
            )
            self.models.append(SimpleNN(40, self.mid_sizes, 1))
            trainer = train.Trainer(
                self.models[k],
                datasets[k],
                loss.MSELoss(),
                epochs,
                batch_size,
                learning_rate,
                learning_rate_decay,
                torch.device(device),
            )
            _, test_loss = trainer.run(save_dir)
            print(f"Stock {k} - Testing Loss = {test_loss[-1]}")
        return

    def nextdayest(self):
        model_out = torch.empty((self.nstocks,))
        for k in range(self.nstocks):
            self.models[k].eval()
            model_out[k] = self.models[k](
                torch.from_numpy(self.data[k, -self.n_before :, :-1].flatten())
                .float()
                .unsqueeze(0)
            )
        return model_out.detach().numpy()
