# ---- voxelmorph ----
# unsupervised learning for image registration
import os
from . import generators
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
backend = os.environ.get('VXM_BACKEND')

if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    from . import torch
    from .torch import layers
    from .torch import networks
    from .torch import losses

elif backend == 'sphere':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "sphere"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    from . import sphere
    from .sphere import layers
    from .sphere import networks
    from .sphere import losses

else:
    # tensorflow is default backend
    try:
        import tensorflow
    except ImportError:
        raise ImportError('Please install tensorflow to use this voxelmorph backend')

    from . import tf
    from .tf import layers
    from .tf import networks
    from .tf import losses
    from .tf import utils
