import pandas as pd
import numpy as np


class FixedSizeEventReader(object):

    def __init__(self, path_to_file, num_events=10000, start_idx=0, drop_last=True):
        print("Read events using fixed size event windows of {} events.".format(num_events))
        print("Output frame rate: variable")

        self.drop_last = drop_last
        self.num_events = num_events

        self._iter = pd.read_csv(path_to_file, delim_whitespace=True, header=None,
                                 names=["t", "x", "y", "pol"],
                                 dtype={"t": np.float64, "x": np.int16, "y": np.int16, "pol": np.int16},
                                 engine="c",
                                 skiprows=start_idx + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __len__(self):
        event_length = len(self._iter.values)
        cluster_length = int(event_length // self.num_events)
        return cluster_length if self.drop_last or event_length % self.num_events == 0 else cluster_length + 1

    def __next__(self):
        event_window = self._iter.__next__().values
        if (len(event_window) == 0) or (len(event_window) < self.num_events and self.drop_last):
            raise StopIteration()
        return event_window


class FixedDurationEventReader(object):

    def __init__(self, path_to_file, duration_ms=50.0, start_idx=0, drop_last=True):
        print("Read events using fixed duration event windows of {} ms.".format(duration_ms))
        print("Output frame rate: {:.1f} Hz".format(1000.0 / duration_ms))

        self.drop_last = drop_last

        self.events = pd.read_csv(path_to_file, delim_whitespace=True, header=None,
                                  names=["t", "x", "y", "pol"],
                                  dtype={"t": np.float64, "x": np.int16, "y": np.int16, "pol": np.int16},
                                  engine="c",
                                  skiprows=start_idx + 1, nrows=None, memory_map=True).values

        self.last_stamp = self.events[0, 0]
        self.duration_s = duration_ms / 1000.

    def __iter__(self):
        return self

    def __len__(self):
        duration_T = self.events[-1, 0] - self.events[0, 0]
        cluster_length = int(duration_T // self.duration_s)
        return cluster_length if self.drop_last else cluster_length + 1

    def __next__(self):
        end_stamp = self.last_stamp + self.duration_s
        event_window = self.events[(self.events[:, 0] >= self.last_stamp) * (self.events[:, 0] < end_stamp)]
        if (len(event_window) == 0) or (end_stamp > self.events[-1, 0] and self.drop_last):
            raise StopIteration()
        self.last_stamp = end_stamp
        return event_window
