EXPERIMENTS = [
    {
        "name": "gelu_layernorm",
        "config_file": "transformer_research/configs/gelu_ln.yaml",
        "norm_type": "layernorm",
        "activation_type": "gelu"
    },
    {
        "name": "gelu_rmsnorm",
        "config_file": "transformer_research/configs/gelu_rms.yaml",
        "norm_type": "rmsnorm",
        "activation_type": "gelu"
    },
    {
        "name": "relu2_layernorm",
        "config_file": "transformer_research/configs/relu2_ln.yaml",
        "norm_type": "layernorm",
        "activation_type": "relu2"
    },
    {
        "name": "relu2_rmsnorm",
        "config_file": "transformer_research/configs/relu2_rms.yaml",
        "norm_type": "rmsnorm",
        "activation_type": "relu2"
    },
    {
        "name": "swiglu_layernorm",
        "config_file": "transformer_research/configs/swiglu_ln.yaml",
        "norm_type": "layernorm",
        "activation_type": "swiglu"
    },
    {
        "name": "swiglu_rmsnorm",
        "config_file": "transformer_research/configs/swiglu_rms.yaml",
        "norm_type": "rmsnorm",
        "activation_type": "swiglu"
    }
]
