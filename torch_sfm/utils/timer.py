from dataclasses import dataclass
from functools import wraps
from time import perf_counter

import torch


@dataclass
class TimingStats:
    count: int = 0
    total: float = 0.0
    last: float = 0.0
    failures: int = 0

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0


class Timer:
    """Wall-clock timing with optional CUDA synchronization and aggregation.

    Use @timer or @timer(verbose=False). Times include nested calls. CUDA is
    synchronized before/after successful calls, without initializing a context
    for CPU work. With device=None, synchronize the current CUDA device.
    """

    def __init__(self, enabled=True, sync_cuda=True, verbose=True, device=None):
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.verbose = verbose
        self.device = torch.device(device) if device is not None else None
        self.stats: dict[str, TimingStats] = {}

    def _synchronize(self):
        if (self.sync_cuda and torch.cuda.is_initialized()
                and (self.device is None or self.device.type == 'cuda')):
            torch.cuda.synchronize(self.device)

    def __call__(self, function=None, *, name=None, verbose=None):
        def decorate(fn):
            label = fn.__qualname__ if name is None else name

            @wraps(fn)
            def wrapped(*args, **kwargs):
                if not self.enabled:
                    return fn(*args, **kwargs)
                self._synchronize()
                start = perf_counter()
                success = False
                try:
                    result = fn(*args, **kwargs)
                    self._synchronize()
                    success = True
                    return result
                finally:
                    # On failure, preserve the original exception rather than
                    # issuing another CUDA call that might hide its cause.
                    elapsed = perf_counter() - start
                    stats = self.stats.setdefault(label, TimingStats())
                    stats.count += 1
                    stats.total += elapsed
                    stats.last = elapsed
                    stats.failures += int(not success)
                    show_output = self.verbose if verbose is None else verbose
                    if show_output:
                        suffix = '' if success else ' (failed)'
                        print(f'[Timer] {label}: {elapsed:.4f} s{suffix}')

            return wrapped

        return decorate if function is None else decorate(function)

    def reset(self):
        self.stats.clear()

    def summary(self):
        """Return cumulative seconds; inclusive totals must not be added together."""
        return {name: {'count': value.count, 'total': value.total, 'mean': value.mean,
                       'last': value.last, 'failures': value.failures}
                for name, value in self.stats.items()}

    def report(self):
        width = max([len(name) for name in self.stats] + [8])
        lines = [f'{"Function":<{width}} {"Calls":>7} {"Total(s)":>12} {"Mean(s)":>12} {"Last(s)":>12}']
        for name, stats in sorted(self.stats.items(), key=lambda item: item[1].total, reverse=True):
            lines.append(f'{name:<{width}} {stats.count:>7} {stats.total:>12.4f} '
                         f'{stats.mean:>12.4f} {stats.last:>12.4f}')
        report = '\n'.join(lines)
        print(report)
        return report
