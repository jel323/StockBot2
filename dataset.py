import numpy as np
import os
import sys

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
        with open(sp_file, "r") as f:
            lines = f.read()
        lines = lines.split("\n")
        lines = [line.split(",") for line in lines]
        self.tickers = [line[0].lower() for line in lines[1:]]
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
        return data, dates

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
