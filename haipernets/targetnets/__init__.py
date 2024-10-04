import haiku as hk
import jax
import jax.numpy as jnp
from .mlp import MLP
from typing import *


def get_mlp(input_dim: int, kwargs: dict, seed: int = 42) -> jax.Array:
    def forward_model(x: jax.Array, kwargs: dict) -> jax.Array:
        module = MLP(**kwargs)
        return module(x)
    
    model = hk.transform(forward_model)
    rng = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, input_dim))
    params = model.init(
        rng,
        dummy_input,
        kwargs
    )
    return model, params, rng


