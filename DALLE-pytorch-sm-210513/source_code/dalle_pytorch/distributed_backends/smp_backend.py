import json
import os

import torch
from importlib import import_module

from .distributed_backend import DistributedBackend



from .distributed_backend import DistributedBackend


class SageMakerMPBackend(DistributedBackend):
    """Distributed backend using the DeepSpeed engine."""

    BACKEND_MODULE_NAME = 'smdistributed.modelparallel.torch'
    BACKEND_NAME = 'SageMakerMP'
    
    def wrap_arg_parser(self, parser):
        return parser


    def _initialize(self, args):
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['LOCAL_RANK'] = str(args.local_rank)


    def has_backend(self):
        try:
            self.backend_module = import_module(self.BACKEND_MODULE_NAME)
        except ModuleNotFoundError:
            return False

        return True

    def wrap_arg_parser(self, parser):
        return parser

    def _initialize(self):
        pass

    def _get_world_size(self):
        return self.backend_module.size()

    def _get_rank(self):
        return self.backend_module.dp_rank()

    def _get_local_rank(self):
        return self.backend_module.local_rank()

    def _local_barrier(self):
        print(f"self.backend_module : {self.backend_module}")
        self.backend_module.barrier()

    def _distribute(
            self,
            _args=None,
            model=None,
            optimizer=None,
            _model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            **_kwargs,
    ):
        """Return the model, optimizer, dataloader, and learning rate scheduler
        as is.
        """
        return (model, optimizer, training_data, lr_scheduler)

    def _average_all(self, tensor):
        averaged = tensor.detach().clone()
        return averaged / self.get_world_size()

