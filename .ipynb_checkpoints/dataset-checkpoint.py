import numpy as np
import os
import sys

import robin_stocks.robinhood as rh

login = rh.login("idm@ianmackey.net", "swimEng97!")

path = sys.path[0]
bpath = path[0 : path.rfind(os.path.sep)]
sys.path.append(bpath)
dpath = os.path.join(bpath, "StockBot", "archive", "Stocks")


def normalize(item):
    mins, spreads, means = np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(len(item)):
        min = np.min(item[i])
        mins[i] = min
        item[i] = item[i] - min

        spread = np.max(item[i])
        spreads[i] = spread
        item[i] = item[i] / spread

        mean = np.mean(item[i])
        means[i] = mean
        item[i] = item[i] - mean
    return mins, spreads, means


def getsptickers(sp_file):
    with open(sp_file, "r") as f:
        lines = f.read()
    lines = lines.split("\n")
    lines = [line.split(",") for line in lines]
    tickers = [line[0].lower() for line in lines[1:]]
    return tickers


def readrh_dict(cdict):
    vals = np.empty((5))
    vals[0] = cdict["open_price"]
    vals[1] = cdict["high_price"]
    vals[2] = cdict["low_price"]
    vals[3] = cdict["close_price"]
    vals[4] = cdict["volume"]
    return vals


def returns(data):
    ret = np.zeros((data.shape[0], data.shape[1] - 1))
    for k in range(data.shape[0]):
        ret[k] = data[k, 1:] / data[k, :-1]
    return ret


class SP_N:
    def __init__(
        self,
        size,
        start_date,
        sp_file="SP500_Index.csv",
        dataroot=dpath,
        normalize=False,
    ):
        # These stocks are naturally sorted by market cap
        self.dataroot = dataroot
        self.start_date = np.datetime64(start_date)
        self.tickers = getsptickers(sp_file)
        self.stocks = np.array(
            [
                ticker + ".us.txt"
                for ticker in self.tickers
                if os.path.isfile(os.path.join(self.dataroot, ticker + ".us.txt"))
            ]
        )
        if size == None:
            self.size = None
        else:
            self.size = np.int64(size)
        self.normalize = normalize
        self.getinds()
        # self.idxs = np.linspace(0, len(self.stocks), self.size + 2).astype(int)
        return

    def getinds(self):
        inds = np.ones((self.__len__(),), np.int64) * -1
        rem = []
        for i, stock in enumerate(self.stocks):
            with open(os.path.join(self.dataroot, self.stocks[i])) as f:
                lines = f.read()
            lines = lines.split("\n")[1:-1]
            lines = np.array([line.split(",") for line in lines])
            dates = lines[:, 0].astype(np.datetime64)
            for j, date in enumerate(dates):
                if date == self.start_date:
                    inds[i] = j
            if inds[i] == -1:
                rem.append(stock)
        stocks2 = []
        inds2 = []
        for i, stock in enumerate(self.stocks):
            if inds[i] != -1:
                stocks2.append(stock)
                inds2.append(inds[i])
        self.stocks = np.array(stocks2)
        self.inds = np.array(inds2, dtype=np.int64)
        return

    def __len__(self):
        return len(self.stocks)

    def __getitem__(self, idx):
        with open(os.path.join(self.dataroot, self.stocks[idx])) as f:
            lines = f.read()
        lines = lines.split("\n")[1:-1]
        lines = np.array([line.split(",") for line in lines])

        dates = lines[:, 0].astype(np.datetime64)
        with open(os.path.join(self.dataroot, self.stocks[idx])) as f:
            lines = f.read()
        lines = lines.split("\n")[1:-1]
        lines = np.array([line.split(",") for line in lines])
        data = np.zeros((self.size, 5), dtype=np.float32)
        data[:, 0] = lines[self.inds[idx] : self.inds[idx] + self.size, 1].astype(
            np.float32
        )
        data[:, 1] = lines[self.inds[idx] : self.inds[idx] + self.size, 2].astype(
            np.float32
        )
        data[:, 2] = lines[self.inds[idx] : self.inds[idx] + self.size, 3].astype(
            np.float32
        )
        data[:, 3] = lines[self.inds[idx] : self.inds[idx] + self.size, 4].astype(
            np.float32
        )
        data[:, 4] = lines[self.inds[idx] : self.inds[idx] + self.size, 5].astype(
            np.float32
        )
        return data

    def getdata(self, nstocks):
        out = np.empty((nstocks, self.size, 5))
        for k in range(nstocks):
            out[k] = self[k]
        return out

    def gettickers(self, nstocks):
        return self.tickers[:nstocks]

    def __getitem2__(self, index):
        sp_stocks = self.stocks[(self.idxs[1:-1] + index) % len(self.stocks)]
        latest_start = None
        earliest_end = None
        stock_tensors = []
        all_dates = []
        for i in range(self.size):
            with open(os.path.join(self.dataroot, sp_stocks[i])) as f:
                lines = f.read()
            lines = lines.split("\n")[1:-1]
            lines = np.array([line.split(",") for line in lines])

            dates = lines[:, 0].astype(np.datetime64)
            all_dates.append(dates)

            opens = lines[:, 1].astype(np.float32)
            highs = lines[:, 2].astype(np.float32)
            lows = lines[:, 3].astype(np.float32)
            closes = lines[:, 4].astype(np.float32)
            volumes = lines[:, 5].astype(np.float32)
            stock_tensors.append(np.array([opens, highs, lows, closes, volumes]))

            if latest_start == None or dates[0] > latest_start:
                latest_start = dates[0]
            if earliest_end == None or dates[-1] < earliest_end:
                earliest_end = dates[-1]

        all_mins, all_spreads, all_means = [], [], []
        for i in range(self.size):
            idxs = np.logical_and(
                (all_dates[i] >= latest_start), (all_dates[i] <= earliest_end)
            ).nonzero()[0]
            stock_tensors[i] = stock_tensors[i][:, idxs]
            if self.normalize:
                mins, spreads, means = normalize(stock_tensors[i])
                all_mins.append(mins)
                all_spreads.append(spreads)
                all_means.append(means)
        stock_tensors = np.stack(stock_tensors)
        if self.normalize:
            all_mins = np.stack(all_mins)
            all_spreads = np.stack(all_spreads)
            all_means = np.stack(all_means)
            return stock_tensors, all_mins, all_spreads, all_means
        else:
            return stock_tensors


def getstockdata(tickers: list, span="5year", interval="day"):
    nstocks = len(tickers)
    vdata = rh.get_stock_historicals(tickers, span=span, interval=interval)
    """if len(vdata) % nstocks != 0:
        raise Exception("Not All Stocks Fill Full 5 years")"""
    ndays = len(vdata) // nstocks
    sdata = np.empty((nstocks, ndays, 5))
    counters = np.zeros(nstocks, dtype=int)
    for i in range(len(vdata)):
        cdict = vdata[i]
        tick = vdata[i]["symbol"]
        tick_idx = tickers.index(tick)

        sdata[tick_idx, counters[tick_idx]] = readrh_dict(cdict)
        counters[tick_idx] += 1

    if ((counters != counters[0]).sum() > 0) and counters[0] == ndays:
        raise Exception("ur wrong")
    return sdata


def getdailydata(rh_data):
    n_points = len(rh_data)
    data_arr = np.empty((n_points, 5))
    for k in range(n_points):
        data_arr[k] = readrh_dict(rh_data[k])
    return np.array(
        [
            data_arr[0, 0],
            data_arr[:, 1].max(),
            data_arr[:, 2].min(),
            data_arr[-1, 3],
            data_arr[:, 4].sum(),
        ]
    )


def upperlst(tickers):
    for k in range(len(tickers)):
        tickers[k] = tickers[k].upper()
    return tickers


class SP_rh:
    def __init__(self, nstocks, sp_file="SP500_Index.csv"):
        self.tickers = upperlst(getsptickers(sp_file)[:nstocks])
        self.stockdata = getstockdata(self.tickers)
        self.addtoday()
        return

    def days(self):
        return self.stockdata.shape[1]

    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):        
        return self.stockdata[idx]

    def getall(self):
        return self.stockdata

    def gettickers(self):
        return self.tickers

    def getcov(self):
        return np.cov(returns(self.stockdata[:, -41:, 3]))

    def addtoday(self):
        stock_data = np.empty((len(self), self.days() + 1, 5))
        stock_data[:, :-1, :] = self.stockdata
        daydata = getstockdata(self.tickers, "day", "5minute")
        daydata = np.array(
            [
                daydata[:, 0, 0],
                daydata[:, :, 1].max(axis=1),
                daydata[:, :, 2].min(axis=1),
                daydata[:, -1, 3],
                daydata[:, :, 4].sum(axis=1),
            ]
        ).T
        stock_data[:, -1, :] = daydata
        self.stockdata = stock_data
        return

    def gettoday(self):
        self.addtoday()
        return self.stockdata[:, -1, :]
