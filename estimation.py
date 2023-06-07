import numpy as np
import matplotlib.pyplot as plt


def getardata1(data, na, nb):
    """
    Used for creating training data
    """
    n = na + nb
    inp = np.zeros((data.shape[0] - n + 1, nb, 4))
    outd = np.zeros((data.shape[0] - n + 1, na))
    for k in range(data.shape[0] - n + 1):
        inp[k] = data[k : k + nb, :-1]
        outd[k] = data[k + nb : k + n, 3]

    inp2 = np.zeros((inp.shape[0], nb * 4))
    for k in range(inp.shape[0]):
        inp2[k] = inp[k].flatten()
    return inp2, outd


def getardata2(data: np.ndarray, nb: int):
    """
    Used for formatting evaluation data, data is given as an np.ndarray, with
    shape - (nstocks, ndays, 4), with last dim being (open, high, low, close)
    """
    out_d = np.empty((data.shape[0], data.shape[1] - nb + 1, 4 * nb))
    for k in range(data.shape[1] - nb + 1):
        out_d[:, k] = data[:, k : k + 10, :].reshape(data.shape[0], 4 * nb)
    return out_d


def addest(input, thta):
    mout = thta @ input.T
    out = np.zeros((input.shape[0], input.shape[1] + 1))
    out[:, :-1] = input
    out[:, -1] = mout
    return out


def makearmodel(
    data: np.ndarray, na=5, nb=10, rho=0.985, c=0.4, type="rls", plotconv=False
):
    input, output = getardata1(data, na, nb)

    thta = []
    inputl = []
    inputl.append(input)
    pc = [plotconv] + [False for k in range(na - 1)]

    for k in range(na):
        if type == "rls":
            thta.append(rlsestimate(inputl[k], output[:, k], rho, c, pc[k]))
        else:
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


class PriceEstimation:
    """
    Initialization parameters
    """

    def __init__(
        self,
        nstocks: int,
        training_data: np.ndarray,
        n_after: int,
        n_before: int,
        type="rls",
        type_params=(),
        make_plots=False,
    ):
        self.nstocks = nstocks
        self.n_after = n_after
        self.n_before = n_before
        self.type = type
        self.type_params = type_params
        self.make_plots = make_plots

        self.makemodel(training_data)
        return

    def makemodel(self, training_data):
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

    def makeestimates(self, data):
        if self.type[-2:] == "ls":
            out = eval_ls(self.model, data, self.n_after, self.n_before)
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
    return thta[-1]


def ls(input, output):
    return np.linalg.inv(input.T @ input) @ input.T @ output


def priceest(data, thta, na=5, nb=10):
    input, output = getardata2(data, nb)
    mout = np.zeros((na, output.shape[0]))
    inputl = [input]
    for k in range(na):
        mout[k] = thta[k] @ inputl[k].T
        inputl.append(addest(inputl[k], thta[k]))
    return mout, output
