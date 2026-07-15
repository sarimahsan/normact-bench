from transformer_research.models.mini_qwen import MiniQwen
from transformer_research.utils.initialization import initialize_weights

MODEL_REGISTRY = {
    "mini_qwen": MiniQwen
}

def build_model(config):
    """
    Builds the model from registry according to configuration and initializes weights.
    """
    model_type = getattr(config, "model_type", "mini_qwen")
    if isinstance(config, dict):
        model_type = config.get("model_type", "mini_qwen")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Available options: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(config)

    # Deterministic weight initialization
    initialize_weights(model, config)

    return model
