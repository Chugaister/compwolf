class ComputingError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class TooManyArgsError(ComputingError):
    def __init__(self):
        super().__init__("too many arguments")


class MaxIterationsError(ComputingError):
    def __init__(self):
        super().__init__("too many iterations")


class ConvergenceError(ComputingError):
    def __init__(self):
        super().__init__("method is not convergence")
