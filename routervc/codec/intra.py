"""Intra. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations


def initialize_intra_decoder_proxy(i_net) -> None:
    """Initialize the released CUDA decoder without running an encoder.

    Upstream creates the DMCI proxy lazily in ``compress`` but assumes it
    already exists in ``decompress``.  A genuinely fresh decoder process has
    no source RGB to perform a warm-up encode, so construct the same proxy
    directly from the frozen state and entropy CDFs.
    """
    from inference_extensions_cuda import DMCIProxy

    state_dict = i_net.add_cdf_to_state_dict(i_net.state_dict())
    i_net.proxy = DMCIProxy()
    i_net.proxy.set_param(state_dict, i_net.gaussian_encoder.skip_thres)
