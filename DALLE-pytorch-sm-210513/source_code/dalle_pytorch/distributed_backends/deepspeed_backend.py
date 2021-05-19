import json
import os

import torch

from .distributed_backend import DistributedBackend


class DeepSpeedBackend(DistributedBackend):
    """Distributed backend using the DeepSpeed engine."""

    BACKEND_MODULE_NAME = 'deepspeed'
    BACKEND_NAME = 'DeepSpeed'

    def wrap_arg_parser(self, parser):
        if not self.has_backend():
            parser.add_argument(
                '--deepspeed',
                type=lambda _: False,
                help=(
                    'whether to use DeepSpeed '
                    "(ignored since it's not available)"
                ),
            )
        else:
            parser = self.backend_module.add_config_arguments(parser)

        parser.add_argument(
            '--local_rank',
            type=int,
            default=-1,
            help='local rank passed from distributed launcher',
        )
        return parser
    
    def sagemaker_init(self, args):
        
        import torch, os
#         torch.distributed.init_process_group(backend="nccl",
#                                              init_method=None)

        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
        args.host_num = args.hosts.index(args.current_host)
        args.local_rank = int(args.rank%args.num_gpus)

        torch.distributed.init_process_group(backend="nccl",
                                rank=args.rank,
                                world_size=args.world_size)
                
        
#         from mpi4py import MPI
#         import subprocess
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#         world_size = comm.Get_size()

#         master_addr = None
#         if rank == 0:
#             hostname_cmd = ["hostname -I"]
#             result = subprocess.check_output(hostname_cmd, shell=True)
#             master_addr = result.decode('utf-8').split()[0]
#         master_addr = comm.bcast(master_addr, root=0)

#         # Determine local rank by assuming hostnames are unique
#         proc_name = MPI.Get_processor_name()
#         all_procs = comm.allgather(proc_name)
#         local_rank = sum([i == proc_name for i in all_procs[:rank]])

        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['LOCAL_RANK'] = str(args.local_rank)
#         os.environ['MASTER_ADDR'] = master_addr
#         os.environ['MASTER_PORT'] = str(distributed_port)  

    def _initialize(self, args):
#         self.backend_module.init_distributed()
        self.sagemaker_init(args)
        if torch.cuda.is_available():
            print(f"self._get_local_rank() : {self._get_local_rank()}")
            torch.cuda.set_device(self._get_local_rank())

    @staticmethod
    def _require_torch_distributed_init():
        """Raise an error when `torch.distributed` has not been
        initialized yet.
        """
        assert torch.distributed.is_initialized(), \
            ('`torch.distributed` is not initialized; please call '
             '`DeepSpeedBackend.initialize` at the start of your script')

    def _get_world_size(self):
        self._require_torch_distributed_init()
        return torch.distributed.get_world_size()

    def _get_rank(self):
        self._require_torch_distributed_init()
        return torch.distributed.get_rank()

    def _get_local_rank(self):
        self._require_torch_distributed_init()
        return int(os.environ['LOCAL_RANK'])

    def _local_barrier(self):
        self._require_torch_distributed_init()
        torch.distributed.barrier()

    def _check_args(self, args, optimizer, lr_scheduler, kwargs):
        """Return an appropriate optimizer and learning rate scheduler
        after checking the values passed to `distribute`.
        """
        self._check_argvs(args, optimizer, lr_scheduler, kwargs)
        (optimizer, lr_scheduler) = self._check_config(
            args, optimizer, lr_scheduler, kwargs)
        return (optimizer, lr_scheduler)

    def _check_argvs(self, args, optimizer, lr_scheduler, kwargs):
        """Apply several sanity checks to the given command
        line arguments.
        """
        has_json_config = (hasattr(args, 'deepspeed_config')
                           and args.deepspeed_config is not None)
        has_dict_config = 'config_params' in kwargs
        if (
                # No config given
                (not has_json_config and not has_dict_config)
                # JSON config file does not exist
                or (not has_dict_config
                    and not os.path.isfile(args.deepspeed_config))
        ):
            # Let DeepSpeed handle these argument errors.
            return

        if not args.deepspeed:
            print(
                'WARNING: DeepSpeed backend was selected; setting '
                '`args.deepspeed = True`'
            )
            args.deepspeed = True

        if has_json_config and has_dict_config:
            print(
                'WARNING: DeepSpeed config was given as both JSON file and '
                'Python dictionary. Python dictionary takes precedence.'
            )

    def _check_config(self, args, optimizer, lr_scheduler, kwargs):
        """Return an appropriate optimizer and learning rate scheduler
        for the DeepSpeed configuration.
        """
        if 'config_params' in kwargs:
            config = kwargs['config_params']
        else:
            with open(args.deepspeed_config, 'r') as json_config_file:
                config = json.load(json_config_file)

        if 'optimizer' in config and optimizer is not None:
            print(
                'WARNING: Optimizer encountered in both DeepSpeed config and '
                'keyword arguments. Optimizer in DeepSpeed config '
                'takes precedence.'
            )
            optimizer = None

        if 'scheduler' in config and lr_scheduler is not None:
            print(
                'WARNING: Learning rate scheduler encountered in both '
                'DeepSpeed config and keyword arguments. Learning rate '
                'scheduler in DeepSpeed config takes precedence.'
            )
            # For the LR scheduler, the JSON config already has
            # precedence. We do this for forward compatibility.
            lr_scheduler = None

        return (optimizer, lr_scheduler)

    def _distribute(
            self,
            args=None,
            model=None,
            optimizer=None,
            model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            **kwargs,
    ):
        """Return a distributed model engine, optimizer, dataloader, and
        learning rate scheduler. These are obtained by wrapping the
        given values with the backend.

        For the other or other possible arguments,
        see `deepspeed.initialize`.
        """
        (optimizer, lr_scheduler) = self._check_args(
            args, optimizer, lr_scheduler, kwargs)

        return self.backend_module.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

    def _average_all(self, tensor):
        self._require_torch_distributed_init()
        # We copy because modification happens in-place
        averaged = tensor.detach().clone()
        # We use `all_reduce` because it is better supported than `reduce`
        torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
        return averaged / self.get_world_size()
