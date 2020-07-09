import time
import atexit

import numpy as np
import torch

timers = {}


class CudaTimer(object):

    def __init__(self, name=""):
        self.name = name
        if self.name not in timers:
            timers[self.name] = []

        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._start.record()
        return self

    def __exit__(self, *args):
        self._end.record()
        torch.cuda.synchronize()
        timers[self.name].append(self._start.elapsed_time(self._end))


class CpuTimer(object):

    def __init__(self, name=""):
        self.name = name

        if self.name not in timers:
            timers[self.name] = []

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._end = time.time()
        self._interval = self._end - self._start
        self._interval *= 1000.0
        timers[self.name].append(self._interval)


def print_timing_info():
    print("-------------- Timing statistics --------------")

    for name, value in [*timers.items()]:
        value = np.mean(np.array(value))
        if value < 1000.0:
            print("{}: {:.2f} ms".format(name, value))
        else:
            print("{}: {:.2f} s".format(name, value / 1000.0))


atexit.register(print_timing_info)


if __name__ == '__main__':
    with CpuTimer("Initialize a tenosr"):
        r = torch.zeros((32, 3, 100, 100), dtype=torch.float64, device="cpu")