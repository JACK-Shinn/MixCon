# coding=utf-8
import math

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MixConConfig(PretrainedConfig):
    model_type = "mixcon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=65536,
            tie_word_embeddings=False,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            num_logits_to_keep=1,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            sliding_window=None,
            max_position_embeddings=262144,
            attention_dropout=0.0,
            num_experts_per_tok=2,
            num_experts=16,
            expert_layer_period=2,
            expert_layer_offset=1,
            attn_layer_period=8,
            attn_layer_offset=4,
            use_conba_kernels=True,
            conba_d_state=16,
            conba_d_conv=4,
            conba_expand=2,
            conba_dt_rank="auto",
            conba_conv_bias=True,
            conba_proj_bias=False,
            learning_rate=1e-4,
            online_learning_rate=1e-5,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self.use_conba_kernels = use_conba_kernels
        self.conba_d_state = conba_d_state
        self.conba_d_conv = conba_d_conv
        self.conba_expand = conba_expand
        self.conba_dt_rank = math.ceil(self.hidden_size / 16) if conba_dt_rank == "auto" else conba_dt_rank
        self.conba_conv_bias = conba_conv_bias
        self.conba_proj_bias = conba_proj_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return [
            "attention" if i % self.attn_layer_period == self.attn_layer_offset else "conba"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layers_num_experts(self):
        return [
            self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
            for i in range(self.num_hidden_layers)
        ]