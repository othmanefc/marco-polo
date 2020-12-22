import pandas as pd


class DataLoader(object):

    def __init__(self, path, chunksize, sep="\t", header=None, names=None):
        self.path = path
        self.chunksize = chunksize
        self.reader = self.load_data(sep, header, names)

    def load_data(self, sep, header, names):
        return pd.read_csv(self.path,
                           sep=sep,
                           chunksize=self.chunksize,
                           header=header,
                           names=names)

    def __len__(self):
        return self.chunksize
