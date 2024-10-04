import jax
import jax.numpy as jnp
import haiku as hk
from typing import *


class MLPHyperNet(hk.Module):
    """
    A hyper-network of MLP architecture which predicts the weights
    of a target neural network.
    """
    def __init__(self, 
        param_shapes: List, 
        hidden_dims: List, 
        num_conditions: int, 
        conditional_emb_dim: int
    ) -> None:
        """
        Parameters:
        -----------
        1. `param_shapes`  :        Shapes of the target weights to predict.
        2. `hidden_dims`   :        Hidden layer dimensions desired in the MLP 
                                    of the hyper-network.
        3. `num_conditons` :        Number of settings to predict the weights in. 
                                    Utilized in an embedding layer which governs 
                                    weight prediction.
        4. `conditional_emb_dim`:   Dimension of the conditional embeddings in the
                                    embedding layer of (shape: `num_conditions`, 
                                    `conditional_emb_dim`).
        """
        self.param_shapes = param_shapes
        self.hidden_dims = hidden_dims
        self.num_conditions = num_conditions
        self.conditional_emb_dim = conditional_emb_dim
        
        self.mapper_dims = [conditional_emb_dim] + hidden_dims if len(hidden_dims) > 0 else []
        self.num_mapper_layers = len(self.mapper_dims)
        self.mapper_activation = jax.nn.gelu
    
    def _get_list_product(self, input_list: List) -> int:
        out = input_list[0]
        for i in range(1, len(input_list)):
            out *= input_list[i]
        return out

    def _reshape_predicted_param(self, param: jax.Array, param_shape: List) -> jax.Array:
        num_conditions = param.shape[0]
        param_shape.insert(0, num_conditions)
        return param.reshape(param_shape)

    def __call__(self, conditions: List) -> jax.Array:
        # define all layers first:

        # 1. embedding layer to model all the conditions
        conditional_lookup = hk.Embed(
            vocab_size=self.num_conditions,
            embed_dim=self.conditional_emb_dim
        )
        # 2. mlp to map the conditional embedding
        mapper_layers = []

        if self.num_mapper_layers == 0:
            mapper_layers.append(jax.lax.identity)
        
        else:
            for i in range(self.num_mapper_layers):
                layer = hk.Linear(self.mapper_dims[i])
                mapper_layers.append(layer)

                if i != self.num_mapper_layers-1:
                    mapper_layers.append(jax.nn.gelu)

        mapper = hk.Sequential(mapper_layers)
        
        # 3. layers to output the parameters of the target model
        #    according to `self.param_shapes`:

        param_predictor = []
        for p_shape in self.param_shapes:
            if len(p_shape) == 1:
                p_dim = p_shape[0]
                predictor = (hk.Linear(p_dim), [-1])
            else:
                p_dim = self._get_list_product(p_shape)
                predictor = (hk.Linear(p_dim), p_shape)
            
            param_predictor.append(predictor)


        # forward pass through the layers:

        # 1. embed the input conditions
        conditions = jnp.array(conditions, dtype=jnp.int32)
        conditional_embedding = conditional_lookup(conditions)

        # 2. map the embedding
        mapped_embedding = mapper(conditional_embedding)

        # 3. predict the parameters
        predicted_params = []
        for predictor in param_predictor:
            param = predictor[0](mapped_embedding)
            param = self._reshape_predicted_param(param, p_shape)
            predicted_params.append(param)

        return predicted_params

