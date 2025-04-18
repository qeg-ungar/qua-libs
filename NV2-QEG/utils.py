import json
import numpy as np


def can_save_json(obj):
    try:
        json.dumps(obj, cls=NumpyEncoder)
        return True
    except TypeError:
        return False


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
