import numpy as np
import matplotlib.pyplot as plt


def getardata1(data, n_after, n_before):
    """
    Used for creating training data
    """
    n = n_after + n_before
    inp = np.zeros((data.shape[0] - n + 1, n_before, 4))
    outd = np.zeros((data.shape[0] - n + 1, n_after))
    for k in range(data.shape[0] - n + 1):
        inp[k] = data[k : k + n_before, :-1]
        outd[k] = data[k + n_before : k + n, 3]

    inp2 = np.zeros((inp.shape[0], n_before * 4))
    for k in range(inp.shape[0]):
        inp2[k] = inp[k].flatten()
    return inp2, outd


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
    input, output = getardata1(data, n_after, n_before)

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


class ARPriceEstimation:
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
        self.trainingdata = training_data
        self.nstocks = self.trainingdata.shape[0]
        self.n_after = n_after
        self.n_before = n_before
        self.type = type
        self.type_params = type_params
        self.make_plots = make_plots

        self.train(training_data)
        return

    def train(self, training_data):
        if self.type[-2:] == "ls":
            self.model = []
            for k in range(self.nstocks):
                self.model.append(
                    makearmodel(
                        training_data[k],
                        self.n_after,
                        self.n_before,
                        self.type_params[0],
                        self.type_params[1],
                        self.type,
                        self.make_plots,
                    )
                )
        return

    def addnewdata(self, ndata):
        narr = np.empty((self.trainingdata.shape[0], self.trainingdata.shape[1] + 1, 5))
        narr[:, :-1] = self.trainingdata
        narr[:, -1] = ndata
        self.trainingdata = narr
        return

    def nextdayest(self):
        model_out = np.empty((self.nstocks))
        for k in range(self.nstocks):
            model_out[k] = (
                self.model[k][0]
                @ self.trainingdata[k, -self.n_before :, :4].flatten().T
            )
        return model_out

    def makeestimates(self, data: np.ndarray):
        if self.type[-2:] == "ls":
            out = eval_ls(self.model, data, self.n_after, self.n_before)
        return out

    def test(self, test_data: np.ndarray, plot=False, slide=False):
        model_out = np.ndarray((self.nstocks, test_data.shape[1]))
        for k in range(test_data.shape[1]):
            model_out[:, k] = self.nextdayest()
            self.addnewdata(test_data[:, k])

        if slide:
            model_out = model_out[:, 1:]
            test_data = test_data[:, :-1, :]

        mean_error = avgerror(test_data[:, :, 3], model_out[:, :], mad)
        print(f"Mean Error - {mean_error}")

        if plot:
            for k in range(self.nstocks):
                plt.figure()
                plt.plot(test_data[k, :60, 3], label="Actual")
                plt.plot(model_out[k, :60], label="Estimate")
                plt.xlabel("Day")
                plt.ylabel("Price")
                plt.legend(loc="lower right")
                plt.show()
        return
