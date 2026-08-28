"""v6 MicroMAEEncoder — exposes `encode(x)` and `decode_from_latent(z)` over an existing AutoEncoder.

Why a separate wrapper instead of editing AutoEncoder.forward_mask
-----------------------------------------------------------------
The base AutoEncoder (`methods/NeighborMix_scMAE/model.py`) only exposes
`forward_mask(x)` returning (latent, mask_logits, reconstruction) — there is no
independent `encode` / `decode_from_latent` pair.  v6 needs both, because the
latent mix step takes two encoder outputs and feeds the result back into the
decoder.

To stay isolated (no edits to the base class), this wrapper:
  1. Receives an existing AutoEncoder instance.
  2. Calls `model.encoder(x)` to get the latent.
  3. Calls `model.mask_predictor(latent)` to get the mask feature.
  4. Calls `model.decoder(cat([latent, mask_feature]))` to get the reconstruction.

This is a pure forward-path wrapper — it does NOT modify the base AutoEncoder
and does NOT introduce any new parameters.

Note
----
The mask feature branch is shared between the latent-mix reconstruction and
the standard forward_mask reconstruction.  When `decoder_use_sigmoid_mask` or
`detach_decoder_mask` differ, the reconstruction may diverge from
`forward_mask`.  We mirror those flags exactly here so the two paths stay
numerically identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MicroMAEEncoder(nn.Module):
    """Lightweight wrapper around AutoEncoder exposing encode / decode_from_latent.

    Args:
        model: an AutoEncoder instance (e.g. the one constructed in run_npz).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # Sanity-check that the wrapped object has the required modules.
        for attr in ("encoder", "mask_predictor", "decoder"):
            if not hasattr(model, attr):
                raise ValueError(
                    f"wrapped model is missing required attribute {attr!r}; "
                    f"expected a scMAE-style AutoEncoder."
                )

    @property
    def hidden_size(self) -> int:
        return int(self.model.hidden_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder(x).  Equivalent to model.feature(x) but kept as a torch
        op so it remains inside the autograd graph (no @torch.no_grad)."""
        self.model._check_expression_shape(x, "x")
        return self.model.encoder(x)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Take latent z and produce the decoder's reconstruction.

        Mirrors the decoder branch of AutoEncoder.forward_mask exactly so that
        the v6 reconstruction is numerically identical to what forward_mask
        would produce given the same latent.
        """
        mask_logits = self.model.mask_predictor(z)
        decoder_mask_feature = mask_logits
        if bool(getattr(self.model, "decoder_use_sigmoid_mask", False)):
            decoder_mask_feature = torch.sigmoid(decoder_mask_feature)
        if bool(getattr(self.model, "detach_decoder_mask", False)):
            decoder_mask_feature = decoder_mask_feature.detach()
        decoder_input = torch.cat([z, decoder_mask_feature], dim=1)
        return self.model.decoder(decoder_input)