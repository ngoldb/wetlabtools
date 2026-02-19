ACTION_REGISTRY = {}

def register_action(name: str):
    def decorator(cls):
        ACTION_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def create_action(tokens: list[str]):
    key = tokens[0].strip().lower()
    args = tokens[1:]

    if key not in ACTION_REGISTRY:
        valid = ", ".join(ACTION_REGISTRY.keys())
        raise ValueError(f"Unknown action '{key}'. Valid actions: {valid}")
    
    cls = ACTION_REGISTRY.get(key)
    try:
        return cls(*args)
    except TypeError as e:
        raise TypeError(
            f"Wrong arguments for {cls.__name__}: got {args}"
        ) from e