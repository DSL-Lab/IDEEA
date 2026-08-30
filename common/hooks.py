def iter_o_proj(model):
    return [(name, module) for name, module in model.named_modules() if "o_proj" in name]


def layer_index(o_proj_name):
    parts = o_proj_name.split(".")
    return int(parts[parts.index("layers") + 1])
