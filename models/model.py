import numpy as np


class Model:
    def __init__(self, model_type, name, created_at, version, metrics=None, model=None):
        self.model_type = model_type
        self.name = name
        self.created_at = created_at
        self.version = version
        self.metrics = metrics
        self.model = model
