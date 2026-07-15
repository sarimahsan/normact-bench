import hashlib
import json

def get_config_hash(config_dict: dict) -> str:
    """
    Generates a deterministic 8-character SHA256 hash based on architectural parameters.
    Ensures that identical configurations map to the same directory path, regardless of order.
    """
    keys_to_hash = [
        "vocab_size", "hidden_dim", "num_layers", "num_heads", "num_kv_heads",
        "max_seq_len", "tie_word_embeddings", "norm_type", "activation_type",
        "dropout", "precision"
    ]
    sub_dict = {k: config_dict[k] for k in keys_to_hash if k in config_dict}
    serialized = json.dumps(sub_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
