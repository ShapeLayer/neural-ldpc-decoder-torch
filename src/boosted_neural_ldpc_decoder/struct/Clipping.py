class Clipping:
    def __init__(
        self,
        abs: float=None,
        start: float=None,
        end: float=None,
    ):
        if abs is None and (start is None or end is None):
            raise ValueError("Either abs or both start and end must be provided")
        
        if abs is not None:
            _sign = 1 if abs >= 0 else -1
            self.start = -abs * _sign
            self.end = abs * _sign
        else:
            self.start = start
            self.end = end
