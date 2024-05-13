import argparse
import importlib
import json
import sys

import torch
import torch.nn as nn

from datasets import node_classification
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator


def get_criterion(task):
    """
    Parameters
    ----------
    task : str
        Name of the task.

    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    if task == 'node_classification':
        criterion = nn.CrossEntropyLoss()

    return criterion

def get_dataset(args):
    """
    Parameters
    ----------
    args : tuple
        Tuple of task, dataset name and other arguments required by the dataset constructor.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        The dataset.
    """
    task, dataset_name, *dataset_args = args
    class_attr = getattr(importlib.import_module('datasets.{}'.format(task)), dataset_name)
    dataset = class_attr(*dataset_args)

    return dataset

def get_agg_class(agg_class):
    """
    Parameters
    ----------
    agg_class : str
        Name of the aggregator class.

    Returns
    -------
    layers.Aggregator
        Aggregator class.
    """
    return getattr(sys.modules[__name__], agg_class)
