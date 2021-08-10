"""
Unit tests for `file_manager.py`
"""


from cc_hdnnp.file_manager import join_paths


def test_join_paths_no_args():
    """
    Test that `join_paths` returns an empty string when no arguments are
    passed.
    """
    assert join_paths() == ""


def test_join_paths_no_slash():
    """
    Test that `join_paths` inserts a slash when required
    """
    assert join_paths("directory", "file") == "directory/file"


def test_join_paths_both_slash():
    """
    Test that `join_paths` removes slashes when required
    """
    assert join_paths("/directory/", "/file/") == "/directory/file/"
