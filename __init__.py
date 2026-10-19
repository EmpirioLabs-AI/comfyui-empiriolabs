"""EmpirioLabs AI nodes for ComfyUI.

ComfyUI imports this file as a package and calls comfy_entrypoint(), which returns the
node list. Test runners may import it on its own, outside a package; then it defines nothing.
"""

if __package__:
    from .empiriolabs_comfy.nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
