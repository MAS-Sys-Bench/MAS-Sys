class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class MASSysError(Exception):
    """Base exception for all MAS-Sys errors"""


class TokenLimitExceeded(MASSysError):
    """Exception raised when the token limit is exceeded"""
