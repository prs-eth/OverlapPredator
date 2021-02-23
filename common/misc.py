"""
Misc utilities
"""

import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys

import coloredlogs
import git


_logger = logging.getLogger()


def print_info(opt, log_dir=None):
    """ Logs source code configuration
    """
    _logger.info('Command: {}'.format(' '.join(sys.argv)))

    # Print commit ID
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y-%m-%d')
        git_message = repo.head.object.message
        _logger.info('Source is from Commit {} ({}): {}'.format(git_sha[:8], git_date, git_message.strip()))

        # Also create diff file in the log directory
        if log_dir is not None:
            with open(os.path.join(log_dir, 'compareHead.diff'), 'w') as fid:
                subprocess.run(['git', 'diff'], stdout=fid)

    except git.exc.InvalidGitRepositoryError:
        pass

    # Arguments
    arg_str = ['{}: {}'.format(key, value) for key, value in vars(opt).items()]
    arg_str = ', '.join(arg_str)
    _logger.info('Arguments: {}'.format(arg_str))


def prepare_logger(opt: argparse.Namespace, log_path: str = None):
    """Creates logging directory, and installs colorlogs

    Args:
        opt: Program arguments, should include --dev and --logdir flag.
             See get_parent_parser()
        log_path: Logging path (optional). This serves to overwrite the settings in
                 argparse namespace

    Returns:
        logger (logging.Logger)
        log_path (str): Logging directory
    """

    if log_path is None:
        if opt.dev:
            log_path = '../logdev'
            shutil.rmtree(log_path, ignore_errors=True)
        else:
            datetime_str = datetime.now().strftime('%y%m%d_%H%M%S')
            if opt.name is not None:
                log_path = os.path.join(opt.logdir, datetime_str + '_' + opt.name)
            else:
                log_path = os.path.join(opt.logdir, datetime_str)

    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler('{}/log.txt'.format(log_path))
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    print_info(opt, log_path)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger, log_path
