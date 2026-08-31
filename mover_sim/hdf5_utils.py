"""Utilities for HDF5-backed scenario output."""


def validate_output_group(output_group):
    """Validate that the provided output target is an `h5py.Group`."""
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("HDF5 output requires h5py to be installed") from exc

    if not isinstance(output_group, h5py.Group):
        raise ValueError("output_group must be an h5py.Group")
    return output_group
