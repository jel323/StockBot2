import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class Portfolio:
    def __init__(
        self,
        start_val,
        stock_tickers,
        days,
        forcast,
        nstocks,
        cov,
        risk_free,
        init_prices,
        gamma=0.002,
        plotp=True,
    ):
        self.start_val = start_val
        self.stock_tickers = np.array(stock_tickers)
        self.days = days
        self.day = 0
        self.forcast = forcast
        self.nstocks = nstocks
        self.risk_free = risk_free
        self.gamma = gamma
        self.plotp = plotp
        self.fixsig(cov)
        self.h = np.zeros((days + 1, nstocks + 1))
        self.h[0, -1] = start_val
        self.hp = np.zeros((days, nstocks + 1))
        self.u = np.zeros((days, nstocks + 1))
        self.z = np.zeros((days, nstocks + 1))
        self.nr = np.zeros((days, nstocks))
        self.n = np.zeros((days, nstocks))
        self.prices = np.zeros((days + 1, nstocks))
        self.prices[0] = init_prices
        self.priceestimates = np.zeros((forcast, nstocks))
        self.returns = np.zeros((days, nstocks + 1))
        self.returns[:, -1] = risk_free * np.ones((days,))
        self.returnestimates = np.zeros((forcast, nstocks + 1))
        self.returnestimates[:, -1] = risk_free * np.ones((forcast,))
        self.maxdret = np.zeros((days,))
        return

    def getvalue(self, day):
        return self.h[day].sum()

    def getposttradevalue(self, day):
        return self.hp[day].sum()

    def getweights(self, day):
        return self.h[day] / self.getvalue(day)

    def grossexposure(self, day):
        return np.sum(np.abs(self.h[day, :-1]))

    def leverage(self, day):
        return self.grossexposure(day) / self.getvalue(day)

    def turnover(self, day):
        return np.sum(np.abs(self.u[day, :-1])) / (2 * self.getvalue(day))

    def transactioncost(self, val):
        return 0

    def holdingcost(self, val):
        return 0

    def randsig(self):
        sigr = np.random.randn(self.nstocks, self.nstocks)
        sigr = sigr.T.dot(sigr)
        sig = np.zeros((self.nstocks + 1, self.nstocks + 1))
        sig[:-1, :-1] = sigr
        return sig

    def fixsig(self, cov):
        sig = np.zeros((self.nstocks + 1, self.nstocks + 1))
        sig[:-1, :-1] = cov
        self.sig = sig
        return

    def calctradesingle(self, plot=True):
        self.returnestimates[0, :-1] = (
            self.priceestimates - self.prices[self.day]
        ) / self.prices[self.day]
        z = cp.Variable(self.nstocks + 1)
        gamma = cp.Parameter(nonneg=True)
        ret = self.returnestimates[0] @ z
        risk = cp.quad_form(z + self.getweights(self.day), self.sig)
        objective = cp.Maximize(ret - gamma * risk)
        constraints = [
            cp.sum(z) == 0,
            z >= -self.getweights(self.day),
            z + self.getweights(self.day) <= 1,
        ]
        prob = cp.Problem(objective, constraints)
        if plot:
            samples = 300
            risk_d = np.zeros((samples,))
            return_d = np.zeros((samples,))
            z_d = np.zeros((samples, self.nstocks + 1))
            gamma_vals = np.logspace(-4, 4, num=samples)
            for k in range(samples):
                gamma.value = gamma_vals[k]
                prob.solve()
                z_d[k] = z.value
                risk_d[k] = cp.sqrt(risk.value).value
                return_d[k] = ret.value + self.returnestimates[0] @ self.getweights(
                    self.day
                )

        k = 0
        gamma.value = self.gamma
        prob.solve()
        return_o = ret.value + self.returnestimates[0] @ self.getweights(self.day)
        risk_o = cp.sqrt(risk.value).value

        print(f"Gamma - {gamma.value}")

        self.z[self.day] = z.value

        if plot:
            plt.figure()
            plt.plot(risk_d, return_d + 1, "g-", label="Efficient Frontier Curve")
            plt.scatter(
                cp.sqrt(np.diagonal(self.sig)).value,
                self.returnestimates[0] + 1,
                c="b",
                label="Individual Stock Risk to Return (Estimate)",
            )
            plt.plot(risk_o, return_o + 1, "k*", label="Chosen Trade Risk to Return")
            plt.xlabel("Risk")
            plt.ylabel("Return")
            plt.legend(loc="lower right")
            plt.show()
        return

    def calctrademulti(self, dropout, plot=True):
        self.returnestimates[0, :-1] = (
            self.priceestimates[0] - self.prices[self.day]
        ) / self.prices[self.day]
        returnestd = self.returnestimates.copy()
        for k in range(1, self.forcast):
            self.returnestimates[k, :-1] = (
                self.priceestimates[k] - self.priceestimates[k - 1]
            ) / self.priceestimates[k - 1]
            returnestd[k, :-1] = (
                dropout**k
                * (self.priceestimates[k] - self.priceestimates[k - 1])
                / self.priceestimates[k - 1]
            )
        z = cp.Variable((self.nstocks + 1, self.forcast))
        ret = cp.trace(returnestd @ z)
        risk_lst = []
        for k in range(self.forcast):
            risk_lst.append(
                dropout**k
                * cp.quad_form(
                    cp.sum(z[:, : k + 1], axis=1) + self.getweights(self.day), self.sig
                )
            )
        risk = cp.sum(risk_lst)
        objective = cp.Maximize(ret - self.gamma * risk)
        constraints = []
        for k in range(self.forcast):
            constraints.append(cp.sum(z[:, k]) == 0)
            constraints.append(
                cp.sum(z[:, : k + 1], axis=1) >= -self.getweights(self.day)
            )
            constraints.append(
                cp.sum(z[:, : k + 1], axis=1) + self.getweights(self.day) <= 1
            )

        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.z[self.day] = z.value[:, 0]
        if plot:
            print(f"Return Matrix - {self.returnestimates @ z.value}")
        return

    def roundp(self, n):
        indp = np.where(n >= 0)
        indn = np.where(n < 0)
        n[indp] = np.floor(n[indp])
        n[indn] = np.ceil(n[indn])
        return n

    def inttrades(self):
        self.u[self.day, :-1] = self.z[self.day, :-1] * self.getvalue(self.day)
        self.nr[self.day] = self.u[self.day, :-1] / self.prices[self.day] + 1e-6
        if self.plotp:
            print(f"Number of each stock to purchase - {self.nr[self.day]}")
        self.n[self.day] = np.floor(self.nr[self.day])
        if self.plotp:
            print(f"Number of each stock to purchase (int) - {self.n[self.day]}")
        nz = np.zeros((self.nstocks + 1,))
        nz[:-1] = self.n[self.day] * self.prices[self.day] / self.getvalue(self.day)
        nz[-1] = -1 * np.sum(nz[:-1])
        if self.plotp:
            print(f"New z with integer stocks - {nz}")
            print(f"Old z - {self.z[self.day]}")
        self.z[self.day] = nz
        return

    def maketrades(self, intt):
        if intt:
            self.inttrades()
        self.u[self.day] = self.z[self.day] * self.getvalue(self.day)
        self.hp[self.day] = self.h[self.day] + self.u[self.day]
        return

    def eststats(self, day):
        estvalv = (
            np.diag(
                np.ones(
                    self.nstocks + 1,
                )
                + self.returnestimates[0]
            )
            @ self.hp[self.day - 1]
        )
        print(f"Estimate Value at t={day} - {estvalv.sum()}")
        return

    def countn(self):
        if self.plotp:
            print(
                f"Total # of Each Stock after {self.day} days - {self.n[:self.day].sum(axis=0)}"
            )
        try:
            assert np.all(self.n[: self.day].sum(axis=0) >= 0)
        except AssertionError:
            print("Negative Number of Single Stock Owned")
            print(
                f"Total # of Each Stock attempted to own - {self.n[:self.day].sum(axis=0)}"
            )
            print(f"Daily Number of Stocks to Buy - {self.nr[self.day - 1]}")
            print(f"Daily Normalized Portfolio Change - {self.z[self.day - 1]}")
            raise
        return

    def truestats(self, day):
        print(f"Value at  - {self.getvalue(self.day)}")
        print(
            f"Daily Return - {100 * self.getvalue(self.day) / self.getvalue(self.day - 1) - 100}%"
        )
        print(f"Total Return - {100 * self.getvalue(self.day) / self.start_val - 100}%")
        print(f"Max Daily Return - {self.maxdret[self.day - 1] * 100}%")
        self.countn()
        print(f"Gross Exposure - {self.grossexposure(self.day)}")
        print(f"Leverage at - {self.leverage(self.day)}")
        print(f"Turnover at - {self.turnover(day - 1)}")
        return

    def testmulti(self):
        returns = self.returns[0:2]
        z = self.z[0:2]
        w = np.zeros((2, self.nstocks + 1))
        w[0] = self.getweights(0)
        w[1] = self.getweights(1)
        print(f"Return for trade 1 - {(returns[0] + 1) @ (w[0] + z[0])}")
        print(f"Return for trade 2 - {(returns[1] + 1) @ (w[1] + z[1])}")
        print(f"Returns - {(returns + 1) @ (z.T + w.T)}")
        return

    def trade(self, new_prices_est, plot=True, intt=True, dropout=1):
        print(f"Day {self.day} - Optimization")
        self.priceestimates = new_prices_est
        if self.forcast == 1:
            self.calctradesingle(plot)
        else:
            self.calctrademulti(dropout)
        self.maketrades(intt)
        self.h[self.day + 1] = (
            np.diag(
                np.ones(
                    self.nstocks + 1,
                )
                + self.returns[self.day]
            )
            @ self.hp[self.day]
        )
        self.day += 1
        print("\n")
        return

    def analyze(self, new_prices_t):
        print(f"Day {self.day} - Analysis")
        self.prices[self.day] = new_prices_t
        self.returns[self.day - 1, :-1] = (
            self.prices[self.day] - self.prices[self.day - 1]
        ) / self.prices[self.day - 1]
        self.maxdret[self.day - 1] = self.returns[self.day - 1].max()
        self.h[self.day] = (
            np.diag(
                np.ones(
                    self.nstocks + 1,
                )
                + self.returns[self.day - 1]
            )
            @ self.hp[self.day - 1]
        )
        self.eststats(self.day)
        self.truestats(self.day)
        print("\n")
        return self.getvalue(self.day), (
            self.getvalue(self.day) / self.getvalue(self.day - 1)
        )

    def printtrades(self):
        ind = self.n[self.day - 1] > 0
        trades_n = self.n[self.day - 1, ind]
        trades_stock = self.stock_tickers[ind]
        for k in range(trades_n.shape[0]):
            print(f"Ticker - {trades_stock[k]}, # Stocks - {trades_n[k]}")
        return
