import numpy as np
import matplotlib.pyplot as plt


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


def addest(input, thta):
    mout = thta @ input.T
    out = np.zeros((input.shape[0], input.shape[1] + 1))
    out[:, :-1] = input
    out[:, -1] = mout
    return out


def getdata(data, na, nb):
    n = na + nb
    ind = np.zeros((data.shape[0] - n + 1, nb, 4))
    outd = np.zeros((data.shape[0] - n + 1, na))
    for k in range(data.shape[0] - n + 1):
        ind[k] = data[k : k + nb, :-1]
        outd[k] = data[k + nb : k + n, 3]

    ind2 = np.zeros((ind.shape[0], nb * 4))
    for k in range(ind.shape[0]):
        ind2[k] = ind[k].flatten()
    return ind2, outd


def makearmodel(data, na=5, nb=10, rho=0.985, c=0.4, type="rls", plotconv=False):
    input, output = getdata(data, na, nb)

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

    return thta, input, output


def priceest(data, thta, na=5, nb=10):
    input, output = getdata(data, na, nb)
    mout = np.zeros((na, output.shape[0]))
    inputl = [input]
    for k in range(na):
        mout[k] = thta[k] @ inputl[k].T
        inputl.append(addest(inputl[k], thta[k]))

    return mout, output
