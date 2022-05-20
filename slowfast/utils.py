import argparse
import functools
import logging
import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd.function import Function
from functools import partial
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from slowfast.config.defaults import get_cfg

_LOCAL_PROCESS_GROUP = None


def init_weights(
        model, fc_init_std=0.01, zero_init_final_bn=True, zero_init_final_conv=False
):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            # Note that there is no bias due to BN
            if hasattr(m, "final_conv") and zero_init_final_conv:
                m.weight.data.zero_()
            else:
                """
                Follow the initialization method proposed in:
                {He, Kaiming, et al.
                "Delving deep into rectifiers: Surpassing human-level
                performance on imagenet classification."
                arXiv preprint arXiv:1502.01852 (2015)}
                """
                c2_msra_fill(m)

        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            if (
                    hasattr(m, "transform_final_bn")
                    and m.transform_final_bn
                    and zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            if hasattr(m, "xavier_init") and m.xavier_init:
                c2_xavier_fill(m)
            else:
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE in {"batchnorm", "sync_batchnorm_apex"}:
        return nn.BatchNorm3d
    elif cfg.BN.NORM_TYPE == "sub_batchnorm":
        return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    elif cfg.BN.NORM_TYPE == "sync_batchnorm":
        return partial(
            NaiveSyncBatchNorm3d, num_sync_devices=cfg.BN.NUM_SYNC_DEVICES
        )
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
        )


class SubBatchNorm3d(nn.Module):
    """
    The standard BN layer computes stats across all examples in a GPU. In some
    cases it is desirable to compute stats across only a subset of examples
    (e.g., in multigrid training https://arxiv.org/abs/1912.00998).
    SubBatchNorm3d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_splits, **args):
        """
        Args:
            num_splits (int): number of splits.
            args (list): other arguments.
        """
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        num_features = args["num_features"]
        # Keep only one set of weight and bias.
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = num_features * num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
        Calculate the aggregated mean and stds.
        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = means.view(n, -1).sum(0) / n
        std = (
                stds.view(n, -1).sum(0) / n
                + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """
        Synchronize running_mean, and running_var. Call this before eval.
        """
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x


class GroupGather(Function):
    """
    GroupGather performs all gather on each of the local process/ GPU groups.
    """

    @staticmethod
    def forward(ctx, input, num_sync_devices, num_groups):
        """
        Perform forwarding, gathering the stats across different process/ GPU
        group.
        """
        ctx.num_sync_devices = num_sync_devices
        ctx.num_groups = num_groups

        input_list = [
            torch.zeros_like(input) for k in range(get_local_size())
        ]
        dist.all_gather(
            input_list, input, async_op=False, group=_LOCAL_PROCESS_GROUP
        )

        inputs = torch.stack(input_list, dim=0)
        if num_groups > 1:
            rank = get_local_rank()
            group_idx = rank // num_sync_devices
            inputs = inputs[
                     group_idx
                     * num_sync_devices: (group_idx + 1)
                                         * num_sync_devices
                     ]
        inputs = torch.sum(inputs, dim=0)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform backwarding, gathering the gradients across different process/ GPU
        group.
        """
        grad_output_list = [
            torch.zeros_like(grad_output) for k in range(get_local_size())
        ]
        dist.all_gather(
            grad_output_list,
            grad_output,
            async_op=False,
            group=_LOCAL_PROCESS_GROUP,
        )

        grads = torch.stack(grad_output_list, dim=0)
        if ctx.num_groups > 1:
            rank = get_local_rank()
            group_idx = rank // ctx.num_sync_devices
            grads = grads[
                    group_idx
                    * ctx.num_sync_devices: (group_idx + 1)
                                            * ctx.num_sync_devices
                    ]
        grads = torch.sum(grads, dim=0)
        return grads, None, None


class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_sync_devices, **args):
        """
        Naive version of Synchronized 3D BatchNorm.
        Args:
            num_sync_devices (int): number of device to sync.
            args (list): other arguments.
        """
        self.num_sync_devices = num_sync_devices
        if self.num_sync_devices > 0:
            assert get_local_size() % self.num_sync_devices == 0, (
                get_local_size(),
                self.num_sync_devices,
            )
            self.num_groups = get_local_size() // self.num_sync_devices
        else:
            self.num_sync_devices = get_local_size()
            self.num_groups = 1
        super(NaiveSyncBatchNorm3d, self).__init__(**args)

    def forward(self, input):
        if get_local_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (
                1.0 / self.num_sync_devices
        )

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias


def cat_all_gather(tensors, local=False):
    """Performs the concatenated all_reduce operation on the provided tensors."""
    if local:
        gather_sz = get_local_size()
    else:
        gather_sz = torch.distributed.get_world_size()
    tensors_gather = [torch.ones_like(tensors) for _ in range(gather_sz)]
    torch.distributed.all_gather(
        tensors_gather,
        tensors,
        async_op=False,
        group=_LOCAL_PROCESS_GROUP if local else None,
    )
    output = torch.cat(tensors_gather, dim=0)
    return output


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def init_process_group(
        local_rank,
        local_world_size,
        shard_id,
        num_shards,
        init_method,
        dist_backend="nccl",
):
    """
    Initializes the default process group.
    Args:
        local_rank (int): the rank on the current local machine.
        local_world_size (int): the world size (number of processes running) on
        the current local machine.
        shard_id (int): the shard index (machine rank) of the current machine.
        num_shards (int): number of shards for distributed training.
        init_method (string): supporting three different methods for
            initializing process groups:
            "file": use shared file system to initialize the groups across
            different processes.
            "tcp": use tcp address to initialize the groups across different
        dist_backend (string): backend to use for distributed training. Options
            includes gloo, mpi and nccl, the details can be found here:
            https://pytorch.org/docs/stable/distributed.html
    """
    # Sets the GPU to use.
    torch.cuda.set_device(local_rank)
    # Initialize the process group.
    proc_rank = local_rank + shard_id * local_world_size
    world_size = local_world_size * num_shards
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        world_size=world_size,
        rank=proc_rank,
    )


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def is_root_proc():
    """
    Determines if the current process is the root process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    Returns:
        (group): pytorch dist group.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    """
    Seriialize the tensor to ByteTensor. Note that only `gloo` and `nccl`
        backend is supported.
    Args:
        data (data): data to be serialized.
        group (group): pytorch dist group.
    Returns:
        tensor (ByteTensor): tensor that serialized.
    """

    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Padding all the tensors from different GPUs to the largest ones.
    Args:
        tensor (tensor): tensor to pad.
        group (group): pytorch dist group.
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device
    )
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_unaligned(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def init_distributed_training(cfg):
    """
    Initialize variables needed for distributed training.
    """
    if cfg.NUM_GPUS <= 1:
        return
    num_gpus_per_machine = cfg.NUM_GPUS
    num_machines = dist.get_world_size() // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == cfg.SHARD_ID:
            global _LOCAL_PROCESS_GROUP
            _LOCAL_PROCESS_GROUP = pg


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


# parse args
def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["configs/Kinetics/SLOWFAST_4x16_R50.yaml"],
        nargs="+",
    )
    parser.add_argument(
        "--opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if is_master_proc() and not os.path.exists(checkpoint_dir):
        try:
            os.mkdir(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


class AllGatherWithGradient(torch.autograd.Function):
    """AllGatherWithGradient"""

    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        x_gather = [torch.ones_like(input) for _ in range(world_size)]
        torch.distributed.all_gather(x_gather, input, async_op=False)
        x_gather = torch.cat(x_gather, dim=0)
        return x_gather

    @staticmethod
    def backward(ctx, grad_output):
        reduction = torch.distributed.all_reduce(grad_output, async_op=True)
        reduction.wait()

        world_size = dist.get_world_size()
        N = grad_output.size(0)
        mini_batchsize = N // world_size
        cur_gpu = torch.distributed.get_rank()
        grad_output = grad_output[
                      cur_gpu * mini_batchsize: (cur_gpu + 1) * mini_batchsize
                      ]
        return grad_output
