

from typing import Union, Any


def manage_params(dictionary: dict[str, Any], param: Union[str, list[str]], default: Any = None):
    if isinstance(param, str):
        param = [param]
    for param in param:
        for param_comp, value in dictionary.items():
            if param in param_comp:
                return value
    return default
