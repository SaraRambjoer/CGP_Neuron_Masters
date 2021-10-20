class Counter:
    def __init__(self) -> None:
        self.count = 0

    def counterval(self):
        self.count += 1
        return self.count
