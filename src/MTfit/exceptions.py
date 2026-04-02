"""
exceptions
==========

Custom exception classes for MTfit.
"""


class MTfitError(Exception):
    """Base exception for all MTfit errors."""


class InversionError(MTfitError):
    """Error during moment tensor inversion."""


class DataParsingError(MTfitError):
    """Error parsing input data files."""


class NoDataError(DataParsingError):
    """No valid data found for inversion."""


class OutputError(MTfitError):
    """Error writing output files."""


class ExtensionError(MTfitError):
    """Error loading or running an extension."""
