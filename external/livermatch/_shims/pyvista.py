def __getattr__(name):  # ofeat: visualisation stub
    raise RuntimeError(f"pyvista.{name} is not available (ofeat shim)")
