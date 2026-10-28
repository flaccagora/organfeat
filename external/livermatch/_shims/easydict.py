class EasyDict(dict):  # ofeat: as defreg/defreg/methods/livermatch.py::_EasyDict
    def __init__(self, d=None, **kw):
        super().__init__()
        for k, v in {**(d or {}), **kw}.items():
            self[k] = v

    def __setitem__(self, k, v):
        super().__setitem__(k, EasyDict(v) if isinstance(v, dict) and not isinstance(v, EasyDict) else v)

    __setattr__ = __setitem__

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e
