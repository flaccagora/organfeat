class SummaryWriter:  # ofeat: no-op logger
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None
