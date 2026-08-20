import numpy as np
import torch
from config import TaskFConfig, ModelConfig
from downstream import EmbeddingEquilibriumSolver


class _Bilinear(torch.nn.Module):
    # V(e1,e2) = e1 . e2 ; saddle of a max_e1 min_e2 game at e1=e2=0 (with bounding).
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        e1, e2 = x[..., : self.d], x[..., self.d:]
        return (e1 * e2).sum(-1)


def test_bounding_keeps_iterates_in_pool_range():
    d = 3
    pool = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    cfg = TaskFConfig(model_config=ModelConfig(model_type="mlp"),
                      outer_steps=50, inner_steps=5, lr_p1=0.5, lr_p2=0.5,
                      num_restarts=1, bound_embeddings=True)
    solver = EmbeddingEquilibriumSolver(_Bilinear(d), pool, pool, cfg, device="cpu")
    res = solver.solve(init_p1=np.array([0.9, 0.9, 0.9]), init_p2=np.array([-0.9, -0.9, -0.9]))
    assert np.all(res.e_p1 <= 1.0 + 1e-5) and np.all(res.e_p1 >= -1.0 - 1e-5)
    assert np.all(res.e_p2 <= 1.0 + 1e-5) and np.all(res.e_p2 >= -1.0 - 1e-5)
    assert len(res.visited) == 50


def test_solver_moves_toward_saddle():
    d = 2
    pool = np.array([[-2.0, -2.0], [2.0, 2.0]])
    cfg = TaskFConfig(model_config=ModelConfig(model_type="mlp"),
                      outer_steps=300, inner_steps=10, lr_p1=0.02, lr_p2=0.1,
                      num_restarts=1, bound_embeddings=True)
    solver = EmbeddingEquilibriumSolver(_Bilinear(d), pool, pool, cfg, device="cpu")
    res = solver.solve(init_p1=np.array([1.5, 1.5]), init_p2=np.array([1.5, 1.5]))
    # near the saddle the value magnitude should be small
    assert abs(res.value) < 1.0
