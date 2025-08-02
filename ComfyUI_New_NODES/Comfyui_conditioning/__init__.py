from .conditioning_cache import SaveConditioningNode, LoadConditioningNode

NODE_CLASS_MAPPINGS = {
    "SaveConditioningNode": SaveConditioningNode,
    "LoadConditioningNode": LoadConditioningNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveConditioningNode": "Save Conditioning",
    "LoadConditioningNode": "Load Conditioning",
}

_all_ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']