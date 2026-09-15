"""Educational 32-rank Data Parallel / ZeRO simulator for ERA V5 Session 12.

The simulator uses one CPU process and models 32 logical GPU ranks. Each logical rank
receives a different data shard. PyTorch autograd computes real per-rank gradients;
the code then explicitly simulates the state ownership and collectives that distinguish
Data Parallel, ZeRO-1, ZeRO-2, and ZeRO-3.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import math
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class ModelShape:
    input_dim: int = 16
    hidden_dim: int = 32
    classes: int = 4

    @property
    def num_parameters(self) -> int:
        return (
            self.input_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.classes
            + self.classes
        )


def init_parameters(shape: ModelShape, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = 0.08
    parts = [
        torch.randn(shape.input_dim * shape.hidden_dim, generator=g) * scale,
        torch.zeros(shape.hidden_dim),
        torch.randn(shape.hidden_dim * shape.classes, generator=g) * scale,
        torch.zeros(shape.classes),
    ]
    return torch.cat(parts).float()


def unpack(flat: torch.Tensor, shape: ModelShape):
    i = 0
    n = shape.input_dim * shape.hidden_dim
    w1 = flat[i:i+n].view(shape.input_dim, shape.hidden_dim); i += n
    b1 = flat[i:i+shape.hidden_dim]; i += shape.hidden_dim
    n = shape.hidden_dim * shape.classes
    w2 = flat[i:i+n].view(shape.hidden_dim, shape.classes); i += n
    b2 = flat[i:i+shape.classes]
    return w1, b1, w2, b2


def forward(flat: torch.Tensor, x: torch.Tensor, shape: ModelShape) -> torch.Tensor:
    w1, b1, w2, b2 = unpack(flat, shape)
    h = torch.tanh(x @ w1 + b1)
    return h @ w2 + b2


def make_dataset(world_size: int, local_batch: int, shape: ModelShape, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    n = world_size * local_batch
    x = torch.randn(n, shape.input_dim, generator=g)
    teacher_w = torch.randn(shape.input_dim, shape.classes, generator=g)
    teacher_b = torch.randn(shape.classes, generator=g) * 0.1
    y = (x @ teacher_w + teacher_b).argmax(dim=-1)
    return [(x[r*local_batch:(r+1)*local_batch], y[r*local_batch:(r+1)*local_batch]) for r in range(world_size)]


def local_gradient(flat: torch.Tensor, batch, shape: ModelShape):
    p = flat.detach().clone().requires_grad_(True)
    x, y = batch
    loss = F.cross_entropy(forward(p, x, shape), y)
    grad, = torch.autograd.grad(loss, p)
    return loss.detach(), grad.detach()


def all_local_gradients(flat: torch.Tensor, rank_batches, shape: ModelShape):
    losses, grads = [], []
    for batch in rank_batches:
        loss, grad = local_gradient(flat, batch, shape)
        losses.append(loss)
        grads.append(grad)
    return torch.stack(losses), grads


def all_local_gradients_threaded(flat: torch.Tensor, rank_batches, shape: ModelShape, max_workers: int = 32):
    """Compute one independent local gradient task per logical rank using CPU threads.

    This is used as an execution proof that the 32 virtual ranks can be backed by
    actual worker threads. The core simulator keeps a sequential path as well so
    notebook results remain deterministic and friendly to constrained runtimes.
    """
    def work(batch):
        return local_gradient(flat, batch, shape)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(work, rank_batches))
    losses = torch.stack([x[0] for x in results])
    grads = [x[1] for x in results]
    return losses, grads


def tensor_shards(x: torch.Tensor, world_size: int) -> List[torch.Tensor]:
    return [part.clone() for part in torch.tensor_split(x, world_size)]


def cat_shards(shards: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(shards) if shards else torch.tensor([])


def average_gradients(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(grads).mean(dim=0)


def reduce_scatter_average(grads: List[torch.Tensor], world_size: int) -> List[torch.Tensor]:
    # Mathematically equivalent to averaging first and then scattering the result.
    return tensor_shards(average_gradients(grads), world_size)


def adam_update(param, grad, m, v, step, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad.square()
    m_hat = m_new / (1 - beta1 ** step)
    v_hat = v_new / (1 - beta2 ** step)
    p_new = param - lr * m_hat / (v_hat.sqrt() + eps)
    return p_new, m_new, v_new


class DataParallelSim:
    """Full parameter, gradient, and optimizer-state replication on every rank."""
    def __init__(self, params, world_size):
        self.world_size = world_size
        self.params = [params.clone() for _ in range(world_size)]
        self.m = [torch.zeros_like(params) for _ in range(world_size)]
        self.v = [torch.zeros_like(params) for _ in range(world_size)]
        self.step_num = 0

    def step(self, rank_batches, shape, lr=1e-2):
        self.step_num += 1
        losses, grads = all_local_gradients(self.params[0], rank_batches, shape)
        avg = average_gradients(grads)  # all-reduce result
        for r in range(self.world_size):
            self.params[r], self.m[r], self.v[r] = adam_update(
                self.params[r], avg, self.m[r], self.v[r], self.step_num, lr=lr
            )
        return losses.mean().item()

    def full_params(self):
        return self.params[0].clone()


class ZeRO1Sim:
    """Optimizer states are sharded; parameters and gradients are logically replicated."""
    def __init__(self, params, world_size):
        self.world_size = world_size
        self.params = [params.clone() for _ in range(world_size)]
        p_shards = tensor_shards(params, world_size)
        self.m = [torch.zeros_like(s) for s in p_shards]
        self.v = [torch.zeros_like(s) for s in p_shards]
        self.step_num = 0

    def step(self, rank_batches, shape, lr=1e-2):
        self.step_num += 1
        losses, grads = all_local_gradients(self.params[0], rank_batches, shape)
        avg = average_gradients(grads)  # each rank can see full averaged gradient
        p_shards = tensor_shards(self.params[0], self.world_size)
        g_shards = tensor_shards(avg, self.world_size)
        new_p = []
        for r in range(self.world_size):
            p, self.m[r], self.v[r] = adam_update(
                p_shards[r], g_shards[r], self.m[r], self.v[r], self.step_num, lr=lr
            )
            new_p.append(p)
        gathered = cat_shards(new_p)  # make updated full weights visible to all ranks
        self.params = [gathered.clone() for _ in range(self.world_size)]
        return losses.mean().item()

    def full_params(self):
        return self.params[0].clone()


class ZeRO2Sim:
    """Optimizer states and averaged gradients are sharded; parameters remain replicated."""
    def __init__(self, params, world_size):
        self.world_size = world_size
        self.params = [params.clone() for _ in range(world_size)]
        p_shards = tensor_shards(params, world_size)
        self.m = [torch.zeros_like(s) for s in p_shards]
        self.v = [torch.zeros_like(s) for s in p_shards]
        self.step_num = 0

    def step(self, rank_batches, shape, lr=1e-2):
        self.step_num += 1
        losses, grads = all_local_gradients(self.params[0], rank_batches, shape)
        grad_shards = reduce_scatter_average(grads, self.world_size)
        p_shards = tensor_shards(self.params[0], self.world_size)
        new_p = []
        for r in range(self.world_size):
            p, self.m[r], self.v[r] = adam_update(
                p_shards[r], grad_shards[r], self.m[r], self.v[r], self.step_num, lr=lr
            )
            new_p.append(p)
        gathered = cat_shards(new_p)
        self.params = [gathered.clone() for _ in range(self.world_size)]
        return losses.mean().item()

    def full_params(self):
        return self.params[0].clone()


class ZeRO3Sim:
    """Parameters, gradients, and optimizer state are persistently sharded."""
    def __init__(self, params, world_size):
        self.world_size = world_size
        self.param_shards = tensor_shards(params, world_size)
        self.m = [torch.zeros_like(s) for s in self.param_shards]
        self.v = [torch.zeros_like(s) for s in self.param_shards]
        self.step_num = 0

    def step(self, rank_batches, shape, lr=1e-2):
        self.step_num += 1
        # Educational simplification: gather the whole tiny model for the forward/backward.
        # Production ZeRO-3/FSDP gathers layer-by-layer and frees promptly.
        transient_full = cat_shards(self.param_shards)
        losses, grads = all_local_gradients(transient_full, rank_batches, shape)
        grad_shards = reduce_scatter_average(grads, self.world_size)
        new_shards = []
        for r in range(self.world_size):
            p, self.m[r], self.v[r] = adam_update(
                self.param_shards[r], grad_shards[r], self.m[r], self.v[r], self.step_num, lr=lr
            )
            new_shards.append(p)
        self.param_shards = new_shards
        del transient_full
        return losses.mean().item()

    def full_params(self):
        # Only for verification / reporting; not persistent ZeRO-3 state.
        return cat_shards(self.param_shards)


def bytes_per_parameter(stage: str, world_size: int) -> float:
    """Session-12 mixed-precision Adam accounting: fp16 weight 2B, fp16 grad 2B,
    fp32 master weight 4B, Adam moments 8B = 16B/parameter before sharding.
    """
    stage = stage.lower()
    if stage in {'dp', 'data parallel', 'data_parallel'}:
        return 16.0
    if stage in {'zero1', 'zero-1', 'zero_1'}:
        return 4.0 + 12.0 / world_size
    if stage in {'zero2', 'zero-2', 'zero_2'}:
        return 2.0 + 14.0 / world_size
    if stage in {'zero3', 'zero-3', 'zero_3'}:
        return 16.0 / world_size
    raise ValueError(stage)


def communication_multiple(stage: str) -> float:
    stage = stage.lower()
    if stage in {'dp', 'data parallel', 'data_parallel', 'zero1', 'zero-1', 'zero_1', 'zero2', 'zero-2', 'zero_2'}:
        return 2.0
    if stage in {'zero3', 'zero-3', 'zero_3'}:
        return 3.0
    raise ValueError(stage)


def theoretical_summary(num_parameters: int, world_size: int) -> List[Dict[str, float]]:
    rows = []
    for name, key in [('Data Parallel','dp'),('ZeRO-1','zero1'),('ZeRO-2','zero2'),('ZeRO-3','zero3')]:
        bpp = bytes_per_parameter(key, world_size)
        rows.append({
            'scheme': name,
            'bytes_per_parameter_per_rank': bpp,
            'memory_bytes_per_rank': bpp * num_parameters,
            'communication_x_P_per_step': communication_multiple(key),
            'optimizer_parameter_fraction_per_rank': 1.0 if key == 'dp' else 1.0/world_size,
            'forward_backward_model_fraction_per_rank': 1.0,
        })
    return rows


def run_equivalence_demo(world_size=32, local_batch=4, steps=3, lr=1e-2, seed=7):
    shape = ModelShape()
    init = init_parameters(shape, seed=seed)
    batches = make_dataset(world_size, local_batch, shape)
    sims = {
        'Data Parallel': DataParallelSim(init, world_size),
        'ZeRO-1': ZeRO1Sim(init, world_size),
        'ZeRO-2': ZeRO2Sim(init, world_size),
        'ZeRO-3': ZeRO3Sim(init, world_size),
    }
    history = []
    for step in range(1, steps+1):
        row = {'step': step}
        for name, sim in sims.items():
            row[name] = sim.step(batches, shape, lr=lr)
        history.append(row)
    ref = sims['Data Parallel'].full_params()
    max_diffs = {name: (sim.full_params() - ref).abs().max().item() for name, sim in sims.items()}
    return shape, init, batches, sims, history, max_diffs


def state_residency_fractions(stage: str, world_size: int) -> Dict[str, float]:
    """Fraction of each logical training-state tensor persistently resident on one rank.

    A value of 1.0 means a full replica. A value of 1/world_size means the state is
    evenly sharded. This is a conceptual ownership view, independent of tensor dtype.
    """
    stage = stage.lower()
    full = 1.0
    shard = 1.0 / world_size
    if stage in {'dp', 'data parallel', 'data_parallel'}:
        return {'parameters': full, 'gradients': full, 'fp32_master': full, 'adam_m': full, 'adam_v': full}
    if stage in {'zero1', 'zero-1', 'zero_1'}:
        return {'parameters': full, 'gradients': full, 'fp32_master': shard, 'adam_m': shard, 'adam_v': shard}
    if stage in {'zero2', 'zero-2', 'zero_2'}:
        return {'parameters': full, 'gradients': shard, 'fp32_master': shard, 'adam_m': shard, 'adam_v': shard}
    if stage in {'zero3', 'zero-3', 'zero_3'}:
        return {'parameters': shard, 'gradients': shard, 'fp32_master': shard, 'adam_m': shard, 'adam_v': shard}
    raise ValueError(stage)


def cluster_redundancy_factor(stage: str, world_size: int) -> float:
    """Total cluster training-state bytes divided by one unique 16 B/parameter copy."""
    return world_size * bytes_per_parameter(stage, world_size) / 16.0


def state_gib_per_rank(stage: str, world_size: int, num_parameters: int) -> float:
    return bytes_per_parameter(stage, world_size) * num_parameters / (1024 ** 3)


def first_fitting_world_size(stage: str, num_parameters: int, capacity_gib: float,
                             candidates=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)):
    """Return the first candidate world size whose analytical training state fits."""
    for n in candidates:
        if state_gib_per_rank(stage, n, num_parameters) <= capacity_gib:
            return n
    return None


def unsynchronized_step_divergence(params: torch.Tensor, grads: List[torch.Tensor], lr: float = 1e-2) -> Tuple[float, List[torch.Tensor]]:
    """Deliberately perform independent rank-local Adam updates without gradient averaging.

    Returns the maximum parameter difference versus rank 0. This is a failure-mode
    experiment: data-parallel replicas diverge when synchronization is removed.
    """
    updates = []
    for grad in grads:
        z = torch.zeros_like(params)
        p, _, _ = adam_update(params.clone(), grad, z, z, step=1, lr=lr)
        updates.append(p)
    ref = updates[0]
    max_div = max((p - ref).abs().max().item() for p in updates[1:]) if len(updates) > 1 else 0.0
    return max_div, updates


def communication_seconds(stage: str, parameter_copy_gb: float, bandwidth_gb_per_s: float) -> float:
    """Idealized transfer time from the Session-12 communication-volume model."""
    return communication_multiple(stage) * parameter_copy_gb / bandwidth_gb_per_s
