import jax
import jax.numpy as jnp
import haiku as hk

from targetnets import get_mlp


def test_mlp():
    input_dim = 3
    kwargs = {
        "hidden_dims": [], 
        "output_dim": 10, 
        "activation": "gelu", 
        "name": "test_mlp"
    }
    seed = 42
    input_tensor = jnp.ones((1, input_dim))

    model, params = get_mlp(input_dim, kwargs, seed=seed)
    print(model)
    print(params)

    output = model.apply(params, seed, input_tensor, kwargs)
    print(output.shape)


if __name__ == "__main__":
    test_mlp()
