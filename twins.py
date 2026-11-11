"""Each of the pack's nodes a second time, under an id only this pack uses.

ComfyUI keeps one node per id. A pack loaded later that also registers H3LivePreview or H3Lora
takes the place of ours, and the editor's renders then fail on inputs that node never had. The
editor's graphs therefore name the twins made here, which no other pack registers.
"""

try:
    from .policy import editor_id
except ImportError:
    from policy import editor_id


def editor_twin(cls):
    """The node class again, registered as editor_id(its id). Hidden from the node menu: it is for the editor's graphs."""
    original = cls.define_schema.__func__

    class Twin(cls):
        # A node class caches these from its schema the first time it is read, so the twin starts clean.
        _DESCRIPTION = None
        _CATEGORY = None
        _DEV_ONLY = None
        SCHEMA = None

        @classmethod
        def define_schema(inner):
            schema = original(inner)
            schema.node_id = editor_id(schema.node_id)
            schema.display_name = "%s (H3 Editor)" % schema.display_name
            schema.is_dev_only = True
            return schema

    Twin.__name__ = Twin.__qualname__ = editor_id(cls.__name__)
    return Twin


def with_twins(classes, names):
    """Node and display-name mappings holding every node and its twin."""
    twins = {editor_id(node_id): editor_twin(cls) for node_id, cls in classes.items()}
    labels = {editor_id(node_id): "%s (H3 Editor)" % names.get(node_id, node_id) for node_id in classes}
    return dict(classes, **twins), dict(names, **labels)
