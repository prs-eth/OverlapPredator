"""PyTorch related utility functions
"""

import logging
import os
import pdb
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


def dict_all_to_device(tensor_dict, device):
    """Sends everything into a certain device """
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device)


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


class CheckPointManager(object):
    """Manager for saving/managing pytorch checkpoints.

    Provides functionality similar to tf.Saver such as
    max_to_keep and keep_checkpoint_every_n_hours
    """
    def __init__(self, save_path: str = None, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0):

        if max_to_keep <= 0:
            raise ValueError('max_to_keep must be at least 1')

        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        self._ckpt_dir = os.path.dirname(save_path)
        self._save_path = save_path + '-{}.pth' if save_path is not None else None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._checkpoints_fname = os.path.join(self._ckpt_dir, 'checkpoints.txt')

        self._checkpoints_permanent = []  # Will not be deleted
        self._checkpoints_buffer = []  # Those which might still be deleted
        self._next_save_time = time.time()
        self._best_score = -float('inf')
        self._best_step = None

        os.makedirs(self._ckpt_dir, exist_ok=True)
        self._update_checkpoints_file()

    def _save_checkpoint(self, step, model, optimizer, score):
        save_name = self._save_path.format(step)
        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'step': step}
        torch.save(state, save_name)
        self._logger.info('Saved checkpoint: {}'.format(save_name))

        self._checkpoints_buffer.append((save_name, time.time()))

        if score > self._best_score:
            best_save_name = self._save_path.format('best')
            shutil.copyfile(save_name, best_save_name)
            self._best_score = score
            self._best_step = step
            self._logger.info('Checkpoint is current best, score={:.3g}'.format(self._best_score))

    def _remove_old_checkpoints(self):
        while len(self._checkpoints_buffer) > self._max_to_keep:
            to_remove = self._checkpoints_buffer.pop(0)

            if to_remove[1] > self._next_save_time:
                self._checkpoints_permanent.append(to_remove)
                self._next_save_time = to_remove[1] + self._keep_checkpoint_every_n_hours * 3600
            else:
                os.remove(to_remove[0])

    def _update_checkpoints_file(self):
        checkpoints = [os.path.basename(c[0]) for c in self._checkpoints_permanent + self._checkpoints_buffer]
        with open(self._checkpoints_fname, 'w') as fid:
            fid.write('\n'.join(checkpoints))
            fid.write('\nBest step: {}'.format(self._best_step))

    def save(self, model: torch.nn.Module, optimizer: Optimizer, step: int, score: float = 0.0):
        """Save model checkpoint to file

        Args:
            model: Torch model
            optimizer: Torch optimizer
            step (int): Step, model will be saved as model-[step].pth
            score (float, optional): To determine which model is the best
        """
        if self._save_path is None:
            raise AssertionError('Checkpoint manager must be initialized with save path for save().')

        self._save_checkpoint(step, model, optimizer, score)
        self._remove_old_checkpoints()
        self._update_checkpoints_file()

    def load(self, save_path, model: torch.nn.Module = None, optimizer: Optimizer = None):
        """Loads saved model from file

        Args:
            save_path: Path to saved model (.pth). If a directory is provided instead, model-best.pth is used
            model: Torch model to restore weights to
            optimizer: Optimizer
        """
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'model-best.pth')

        state = torch.load(save_path)

        step = 0
        if 'step' in state:
            step = state['step']

        if 'state_dict' in state and model is not None:
            model.load_state_dict(state['state_dict'])

        if 'optimizer' in state and optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])

        self._logger.info('Loaded models from {}'.format(save_path))
        return step


class TorchDebugger(torch.autograd.detect_anomaly):
    """Enters debugger when anomaly detected"""
    def __enter__(self) -> None:
        super().__enter__()

    def __exit__(self, type, value, trace):
        super().__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            print(value)
            if sys.gettrace() is None:
                pdb.set_trace()
