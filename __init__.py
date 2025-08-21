"""
MoMask Codes Package
Human Motion Generation and Editing with Masked Transformers

This package contains the complete implementation of the MoMask model for
text-to-motion generation and motion editing tasks.
"""

__version__ = "0.0.0"
__author__ = "EricGuo"

# Main model exports
from .models.vq.model import RVQVAE, LengthEstimator
from .models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from .models.loaders import load_vq_model, load_trans_model, load_res_model, load_len_estimator

# Data loading utilities
from .motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from .data.t2m_dataset import Text2MotionDataset, MotionDataset

# Evaluation and utilities
from .models.t2m_eval_wrapper import EvaluatorModelWrapper
from .utils.get_opt import get_opt
from .utils.fixseed import fixseed
from .utils.motion_process import recover_from_ric
from .utils.plot_script import plot_3d_motion

# Options and configuration
from .options.eval_option import EvalT2MOptions
from .options.train_option import TrainT2MOptions
from .options.vq_option import arg_parse

# Visualization
from .visualization.joints2bvh import Joint2BVHConvertor

__all__ = [
    # Models
    'RVQVAE', 'LengthEstimator', 'MaskTransformer', 'ResidualTransformer',
    'load_vq_model', 'load_trans_model', 'load_res_model', 'load_len_estimator',
    
    # Data
    'get_dataset_motion_loader', 'Text2MotionDataset', 'MotionDataset',
    
    # Evaluation
    'EvaluatorModelWrapper',
    
    # Utilities
    'get_opt', 'fixseed', 'recover_from_ric', 'plot_3d_motion',
    
    # Options
    'EvalT2MOptions', 'TrainT2MOptions', 'arg_parse',
    
    # Visualization
    'Joint2BVHConvertor',
]
