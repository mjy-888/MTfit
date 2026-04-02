"""
extensions.py
******************

Utility functions for handling MTfit extensions.

Simple functions that are used throughout the module
"""


# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.


from importlib.metadata import entry_points


def get_extensions(group: str, defaults: dict | bool = False) -> tuple[list[str], dict]:
    """
    Get the installed extensions for a given entry point group.

    Args
        group: entry point group name.

    Keyword Args
        defaults: dictionary of name:function pairs for default values.

    Returns
        (list, dict): tuple of extension name list and dictionary of extension name : function pairs.
    """
    names = []
    funcs = {}
    if isinstance(defaults, dict):
        for plugin_name, plugin in defaults.items():
            if plugin_name not in names:
                funcs[plugin_name] = plugin
                names.append(plugin_name)
    for ep in entry_points(group=group):
        plugin = ep.load()
        names.append(ep.name.lower())
        funcs[ep.name.lower()] = plugin
    names = list(set(names))
    return (names, funcs)


def evaluate_extensions(group: str, defaults: dict | bool = False, **kwargs) -> list:
    """
    Return the list of results from evaluating all the extensions in a given group.

    Args
        group: entry point group name.

    Keyword Args
        defaults: dictionary of name:function pairs for default values.

    Returns
        list: List of results from evaluating each extension's function.
    """
    results = []
    try:
        if isinstance(defaults, dict):
            for plugin in defaults.values():
                results.append(plugin(**kwargs))
        for ep in entry_points(group=group):
            try:
                plugin = ep.load()
                results.append(plugin(**kwargs))
            except Exception:
                pass
    except Exception:
        pass
    return results
