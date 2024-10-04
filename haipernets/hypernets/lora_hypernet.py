import jax
import jax.numpy as jnp
import haiku as hk
from typing import *


class MLPHyperNet(hk.Module):
    def __init__(self, 
        param_shapes: List, 
        hidden_dims: List, 
        num_conditions: int, 
        conditional_emb_dim: int,
        lora_rank: int
    ) -> None:
        self.param_shapes = param_shapes
        self.hidden_dims = hidden_dims
        self.num_conditions = num_conditions
        self.conditional_emb_dim = conditional_emb_dim
        self.lora_rank = lora_rank
        
        self.mapper_dims = [conditional_emb_dim] + hidden_dims if len(hidden_dims) > 0 else []
        self.num_mapper_layers = len(self.mapper_dims)
        self.mapper_activation = jax.nn.gelu
    
    def _get_list_product(self, input_list: List) -> int:
        assert len(input_list) == 2, "Error: 3 dimensional parameter breaks this LoRA-based weight prediction!"
        out = input_list[0]
        for i in range(1, len(input_list)):
            out *= input_list[i]
        return out

    def _get_list_matmul(self, input_list: List) -> int:
        assert len(input_list) == 2, "Error: 3 dimensional parameter breaks this LoRA-based weight prediction!"
        out = input_list[0]
        for i in range(1, len(input_list)):
            out = out @ input_list[i].T
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
            assert len(p_shape) >= 1, "Error: empty parameter shape given to hyper-network!"
            if len(p_shape) == 1:
                p_dim = p_shape[0]
                predictor = (hk.Linear(p_dim), [-1])
            else:
                decomposition_predictor = []
                for dim in p_shape:
                    decomposition_predictor.append(hk.Linear(dim * self.lora_rank))    
                predictor = (decomposition_predictor, p_shape)
            
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
            if len(predictor[1]) == 1: 
                param = predictor[0](mapped_embedding)
                param = self._reshape_predicted_param(param, p_shape)
            
            else:
                p_shape = predictor[1]
                param_decomposition = [ decomp_predictor(mapped_embedding) for decomp_predictor in predictor[0] ]
                param_decomposition = [ self._reshape_predicted_param(param, [dim, self.lora_rank]) for param, dim in zip(param_decomposition, p_shape)]
                param = self._get_list_matmul(param_decomposition)
            
            predicted_params.append(param)    

        return predicted_params
