"""
Utility scripts for handling files.
"""


def join_paths(*paths: str):
    """
    Join arguments into a single file path. If there isn't a trailing or
    leading slash, for the strings, one is added. If both are present, one is
    removed before concatenation.

    Parameters
    ----------
    *paths : str
        Two or more strings to be joined into one filepath.

    Returns
    -------
    str
        The complete filepath.
    """

    complete_path = ''
    for i, path in enumerate(paths):
        if i > 0:
            if path[0] == '/':
                complete_path += path
            else:
                complete_path += '/' + path

        if path[-1] == '/':
            complete_path += path[:-1]
        else:
            complete_path += path[:-1]
    
    return complete_path