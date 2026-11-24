import sys
import warnings


def _configure_known_warning_filters() -> None:
    """Filter a small set of known third-party warnings that add noise at startup."""
    warnings.filterwarnings(
        "ignore",
        message=r"(?s)Valid config keys have changed in V2:.*underscore_attrs_are_private.*",
        category=UserWarning,
        module=r"pydantic\._internal\._config",
    )

    warnings.filterwarnings(
        "ignore",
        message=r".*doesn't match a supported version!$",
        category=Warning,
        module=r"requests",
    )


_configure_known_warning_filters()


if sys.version_info[:2] < (3, 11) or sys.version_info[:2] > (3, 13):
    print(
        "Warning: Unsupported Python version {ver}, please use 3.11-3.13".format(
            ver=".".join(map(str, sys.version_info))
        )
    )
