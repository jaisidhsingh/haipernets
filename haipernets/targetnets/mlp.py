import haiku as hk
import jax
import jax.numpy as jnp
from typing import *


class MLP(hk.Module):
    def __init__(self, hidden_dims: List, output_dim: int, activation: str = "gelu", name: str = None) -> None:
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = len(hidden_dims) + 1
        self.activation = activation
    
    def __call__(self, x: jax.Array) -> jax.Array:
        layers = []
        layer_dims = self.hidden_dims + [self.output_dim]

        for i in range(self.num_layers):
            learnable_layer = hk.Linear(layer_dims[i])
            layers.append(learnable_layer)

            if i != self.num_layers-1:
                if self.activation == "gelu":
                    layers.append(jax.nn.gelu)
                elif self.activation == "relu":
                    layers.append(jax.nn.relu)
            
        mlp = hk.Sequential(layers)
        return mlp(x)
