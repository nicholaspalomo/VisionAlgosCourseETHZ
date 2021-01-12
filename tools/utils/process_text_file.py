import numpy as np

class ProcessTextFile:
    def __init__(self):
        self.fname_ = ""
        self.file_contents_ = []

    def read_file(self, fname, delimiter=None):
        self.fname_ = fname
        self.file_contents_ = np.loadtxt(self.fname_, dtype=np.double, comments='#', delimiter=delimiter)

        return self.file_contents_.copy()

    def write_file(self):

        return

    def close(self):

        return

    def read(self):

        return 