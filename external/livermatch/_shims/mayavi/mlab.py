def __getattr__(name):  # ofeat: visualisation stub
    raise RuntimeError(f"mayavi.mlab.{name} is not available (ofeat shim)")
