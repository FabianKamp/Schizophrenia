import time
class TimerError(Exception):
    """
    Exception error to report timer errors.
    """

class timer:
    def __init__(self):
        self._tic = None

    def tic(self):
        if self._tic is not None:
            raise TimerError('Timer is already running.')

        self._tic = time.perf_counter()

    def toc(self):
        if self._tic is None:
            raise TimerError('Timer must be started using tic before it is ended using toc.')
        elapsed_time = time.perf_counter() - self._tic
        self._tic = None

        print(f'Elapsed time: {elapsed_time:0.4f} seconds')
        return elapsed_time