class Shortening:
    def __init__(
        self,
        start: int,
        end: int,
    ):
        self.start = start
        self.end = end
        if self.start < 0 or self.end < 0 or self.start > self.end:
            raise ValueError("Invalid shortening range")
        self._len = self.end - self.start + 1
        
    def __len__(self):
        return self._len
