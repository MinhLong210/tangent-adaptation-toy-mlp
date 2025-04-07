import abc
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Callable, List

import torch
import torch.nn as nn
from torch import Tensor
from peft.tuners.lora import LoraLayer

# from functorch import make_functional_with_buffers
from torch.func import functional_call, jvp

log = logging.getLogger(__name__)


def dict_params_to_tuple(dict_params: dict):
    return tuple(v for k, v in dict_params.items())


class LinearizedModelWraper(nn.Module):
    def __init__(self, model: nn.Module, init_model: nn.Module = None):
        """
        Initializes a linearized model.

        Args:
            model (nn.Module): The underlying PyTorch model to be linearized.
            init_model (nn.Module): The initial PyTorch model used to compute the linearization parameters (default: None).
        """
        super().__init__()
        self.model = model
        if init_model is None:
            init_model = model
        assert not hasattr(self, "params0")
        params0 = deepcopy([(k, v.detach()) for k, v in init_model.named_parameters()])
        self.params0_keys = [k for k, v in params0]
        self.params0_values = nn.ParameterList([v for k, v in params0])
        for p in self.params0_values:
            p.requires_grad_(False)

    def tuple_params_to_dict(self, tuple_params):
        """
        Converts a tuple of parameters to a dictionary with keys corresponding to the parameter names.

        Args:
            tuple_params (Tuple[Tensor, ...]): A tuple of parameters.

        Returns:
            Dict[str, Tensor]: A dictionary with keys corresponding to the parameter names and values corresponding to the
            parameter values.
        """
        assert len(tuple_params) == len(self.params0_keys)
        state_dict = {}
        for k, p in zip(self.params0_keys, tuple_params):
            state_dict[k] = p
        return state_dict

    def forward(self, *args, **kwargs):
        """
        Computes the linearized model output using a first-order Taylor decomposition.

        Args:
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.

        Returns:
            torch.Tensor: The output of the linearized model, computed using a first-order Taylor decomposition.
        """
        params0 = tuple(self.params0_values)
        params = dict_params_to_tuple(OrderedDict(self.named_parameters()))
        dparams = tuple(p - p0 for p, p0 in zip(params, params0))
        out, dp = jvp(
            lambda *param: functional_call(
                self.model, self.tuple_params_to_dict(param), args, kwargs
            ),
            params0,
            dparams,
        )
        return out + dp


def _get_submodules(model: nn.Module, key):
    """
    Retrieves the parent module, target module, and target module name for a given key in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to retrieve submodules from.
        key (str): The key representing the submodule to retrieve.

    Returns:
        Tuple[nn.Module, nn.Module, str]: A tuple containing the parent module, target module, and target module name.
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def linearize_lora_model(model: nn.Module):
    """
    Linearizes the LoraLayer modules in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be linearized.

    Returns:
        nn.Module: The linearized PyTorch model.
    """
    for key, module in model.named_modules():
        if isinstance(module, LoraLayer):
            print(f"convert {key} to linearized lora layer")
            parent, target, target_name = _get_submodules(model, key)
            setattr(parent, target_name, LinearizedModelWraper(target))
    return model