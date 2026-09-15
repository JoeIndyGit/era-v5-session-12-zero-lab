import torch
from src.zero_sim import (
    ModelShape, init_parameters, tensor_shards, cat_shards,
    run_equivalence_demo, bytes_per_parameter, make_dataset,
    all_local_gradients, all_local_gradients_threaded,
    average_gradients, reduce_scatter_average, state_residency_fractions,
    cluster_redundancy_factor, first_fitting_world_size,
    unsynchronized_step_divergence, communication_seconds,
)


def test_shard_round_trip_with_uneven_split():
    # 676 is deliberately not divisible by 32: real sharding code must handle a tail.
    x = torch.arange(676, dtype=torch.float32)
    shards = tensor_shards(x, 32)
    assert torch.equal(cat_shards(shards), x)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_memory_formulas_at_32_ranks():
    assert bytes_per_parameter('dp', 32) == 16.0
    assert abs(bytes_per_parameter('zero1', 32) - 4.375) < 1e-12
    assert abs(bytes_per_parameter('zero2', 32) - 2.4375) < 1e-12
    assert abs(bytes_per_parameter('zero3', 32) - 0.5) < 1e-12


def test_collective_identity():
    shape = ModelShape()
    p = init_parameters(shape)
    batches = make_dataset(32, 2, shape)
    _, grads = all_local_gradients(p, batches, shape)
    all_reduce = average_gradients(grads)
    rs_then_ag = cat_shards(reduce_scatter_average(grads, 32))
    assert torch.allclose(all_reduce, rs_then_ag)


def test_zero_stages_match_data_parallel_update():
    _, _, _, _, _, diffs = run_equivalence_demo(world_size=32, local_batch=2, steps=3, lr=1e-2)
    for name, diff in diffs.items():
        assert diff < 1e-6, (name, diff)


def test_threaded_32_rank_execution_matches_sequential():
    torch.set_num_threads(1)
    shape = ModelShape()
    p = init_parameters(shape)
    batches = make_dataset(32, 2, shape)
    loss_a, grad_a = all_local_gradients(p, batches, shape)
    loss_b, grad_b = all_local_gradients_threaded(p, batches, shape, max_workers=32)
    assert torch.allclose(loss_a, loss_b)
    for a, b in zip(grad_a, grad_b):
        assert torch.allclose(a, b)


def test_state_ownership_progressively_shards_more_state():
    dp = state_residency_fractions('dp', 32)
    z1 = state_residency_fractions('zero1', 32)
    z2 = state_residency_fractions('zero2', 32)
    z3 = state_residency_fractions('zero3', 32)
    assert dp['adam_m'] == 1.0 and z1['adam_m'] == 1/32
    assert z1['gradients'] == 1.0 and z2['gradients'] == 1/32
    assert z2['parameters'] == 1.0 and z3['parameters'] == 1/32
    assert cluster_redundancy_factor('dp', 32) > cluster_redundancy_factor('zero1', 32) > cluster_redundancy_factor('zero2', 32) > cluster_redundancy_factor('zero3', 32)
    assert cluster_redundancy_factor('zero3', 32) == 1.0


def test_30b_fit_boundaries_match_session_12_state_model():
    p = 30_000_000_000
    capacity = 74.5
    assert first_fitting_world_size('dp', p, capacity) is None
    assert first_fitting_world_size('zero1', p, capacity) is None
    assert first_fitting_world_size('zero2', p, capacity) == 32
    assert first_fitting_world_size('zero3', p, capacity) == 8


def test_synchronization_is_required_for_data_parallel_equivalence():
    shape = ModelShape()
    p = init_parameters(shape)
    batches = make_dataset(32, 2, shape)
    _, grads = all_local_gradients(p, batches, shape)
    divergence, _ = unsynchronized_step_divergence(p, grads)
    assert divergence > 1e-6


def test_communication_time_model():
    # 30B fp16 parameters => P=60 GB. At 50 GB/s: 2P=2.4 s, 3P=3.6 s.
    assert abs(communication_seconds('zero2', 60.0, 50.0) - 2.4) < 1e-12
    assert abs(communication_seconds('zero3', 60.0, 50.0) - 3.6) < 1e-12
