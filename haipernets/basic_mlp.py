import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class MyLinearClass(hk.Module):
    def __init__(self, output_dim, name=None):
        super().__init__(name=name)
        self.output_dim = output_dim
    
    def __call__(self, x):
        input_dim = x.shape[-1]
        
        # initalize weight
        w_init = hk.initializers.TruncatedNormal(1.0 / jnp.sqrt(input_dim))
        w = hk.get_parameter("w", shape=[input_dim, self.output_dim], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[self.output_dim], dtype=x.dtype, init=jnp.ones)
        return x @ w + b


def forward_mlp(x):
    mlp = hk.Sequential([
        hk.Linear(50),
        jax.nn.gelu,
        hk.Linear(10)
    ])
    return mlp(x)


mlp = hk.transform(forward_mlp)

input_tensor = jnp.ones((1, 3))
seed = jax.random.PRNGKey(32)
params = mlp.init(rng=seed, x=input_tensor)

output = mlp.apply(params=params, x=input_tensor, rng=seed)
print(output.shape)

print(jax.devices())
