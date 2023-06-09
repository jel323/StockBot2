import numpy as np
import dataset as dset
import estimation2 as est
import portfolio as pf


class Sim:
    """
    This class is used to simulate how the trading would go for the last self.days days of
        a given stock dataset
    """

    def __init__(
        self,
        startval,
        stock_tickers,
        dataset: np.ndarray,
        sim_days: int,
        n_after: int,
        n_before: int,
        gamma=200,
        plot_f=True,
    ):
        self.start_val = startval
        self.stock_tickers = stock_tickers
        self.sim_days = sim_days
        self.nstocks = len(self.stock_tickers)
        self.trainingdata = dataset[:, :-sim_days]
        self.testingdata = dataset[:, -sim_days:]
        self.n_after = n_after
        self.n_before = n_before
        self.gamma = gamma
        self.pestimates = np.empty((self.sim_days + 1, self.nstocks))
        self.plot_f = plot_f
        return

    def setup(self):
        self.estimator = est.ARPriceEstimation(
            self.nstocks, self.trainingdata, self.n_after, self.n_before, "rls"
        )
        self.cov = np.cov(dset.returns(self.trainingdata[:, -41:, 3]))
        self.portfolio = pf.Portfolio(
            self.start_val,
            self.stock_tickers,
            self.sim_days + 1,
            self.n_after,
            self.nstocks,
            self.cov,
            0.0001,
            self.trainingdata[:, -1, 3],
            self.gamma,
        )
        self.pestimates[0] = self.estimator.nextdayest()
        self.portfolio.trade(self.pestimates[0], self.plot_f)
        self.portfolio.printtrades()
        return

    def step(self, newdata, day):
        self.portfolio.analyze(newdata[:, 3])
        self.estimator.newdaydata(newdata)
        self.estimator.trainnewday()
        self.pestimates[day] = self.estimator.nextdayest()
        self.portfolio.trade(self.pestimates[day], self.plot_f)
        return

    def sim(self):
        for k in range(self.sim_days):
            self.step(self.testingdata[:, k], k + 1)
        return
