# `haipernets`

A hyper-network library written in <a href="https://dm-haiku.readthedocs.io/en/latest/index.html">`haiku`</a> and <a href="https://jax.readthedocs.io/en/latest/">`jax`</a>.

## Requirements
The `jax` and `haiku` libraries are all that you need. These can be installed by simply running the following:

```bash
pip3 install jax
pip3 install dm-haiku
```

## Features / Todo

The following features have been integrated into the library with more on the way.

### Hypernetwork architectures
- [x] Linear
- [x] MLP-based
- [x] LoRA-based

### Target network architectures
- [x] MLP


## Usage

The library is meant to reduce re-writing of redundant code at the user's end, and wraps all `haiku` transformations of provided networks internally. All you have to do is import and go!

```python
import jax.numpy as jnp
from haipernets.targetnets import get_mlp
from haipernets.hypernets import get_lora_hypernet


# An example config
input_dim = 768
batch_size = 16
rng_seed = 42
mlp_config = {
    "hidden_dims" [],
    "output_dim": 384,
    "activation": "relu",
    "name": "my_linear_layer"
}

# Load in an MLP as the target network. 
# Here it's just a linear layer (`mlp_config['hidden_dims'] = []`) 
mlp, mlp_params, rng = get_mlp(input_dim, mlp_config, rng_seed)

# Test the target network's forward pass 
input_tensor = jnp.ones((batch_size, input_dim))
output = mlp.apply(params=mlp_params, x=input_tensor, rng=rng)


# Load in a hypernetwork to predict the weights of the target network.
# Here it's a LoRA-based hypernetwork architecture of small parametric cost.
# TODO

```