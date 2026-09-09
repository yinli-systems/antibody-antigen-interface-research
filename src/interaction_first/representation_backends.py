"""Concrete encoders for :mod:`interaction_first.pretrained_representations`.

Three encoders, chosen because published evidence attributes a specific gain to
each and none of them can see inter-chain geometry:

``antiberty``
    Antibody-specific masked language model, 26M parameters over 558M OAS
    heavy/light sequences. DeepAAAssembly's feature-relevance analysis names it
    one of the two features that matter most for characterising antibody-antigen
    interaction patterns.

``esm2``
    General protein language model, for parity with the representations already
    stored under the ``sppider_seq_embedding_v1`` format and as the control
    against which an antibody-specific model has to justify itself.

``esm_if1``
    Inverse-folding encoder over a backbone. DiscoTope-3.0 attributes its margin
    over BepiPred-3.0 and ScanNet to these latents. It reads one chain's own
    backbone only, so it is intra-chain by construction.

Every backend pins the checkpoint it loaded and reports it in the receipt, and
every import is deferred to construction time so this module can be imported,
inspected, and tested on a machine with none of the weights.
"""

from __future__ import annotations

from typing import Any

from .pretrained_representations import RepresentationError, RepresentationUnit


class _TorchBackend:
    """Shared device handling and no-grad inference."""

    requires_backbone = False

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model: Any = None

    def _torch(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RepresentationError("PyTorch is required") from error
        return torch


class AntiBertyBackend(_TorchBackend):
    """AntiBERTy per-residue embeddings for an antibody chain."""

    name = "antiberty"
    dim = 512
    revision = "antiberty-pypi"

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                from antiberty import AntiBERTyRunner
            except ModuleNotFoundError as error:
                raise RepresentationError(
                    "antiberty is not installed (pip install antiberty)"
                ) from error
            self._model = AntiBERTyRunner()
        return self._model

    def encode(self, unit: RepresentationUnit) -> Any:
        torch = self._torch()
        runner = self._ensure()
        with torch.no_grad():
            embeddings = runner.embed([unit.sequence])
        # AntiBERTyRunner returns [1 + length + 1, dim] with BOS/EOS retained.
        residues = embeddings[0]
        if residues.shape[0] == len(unit.sequence) + 2:
            residues = residues[1:-1]
        if residues.shape[0] != len(unit.sequence):
            raise RepresentationError(
                f"antiberty returned {residues.shape[0]} rows for a "
                f"{len(unit.sequence)}-residue sequence"
            )
        return residues.to(torch.float32).cpu()


class Esm2Backend(_TorchBackend):
    """ESM-2 per-residue embeddings from the final representation layer."""

    name = "esm2_t33_650M_UR50D"
    dim = 1280
    revision = "facebook/esm2_t33_650M_UR50D"

    def __init__(self, device: str = "cpu", model_name: str | None = None) -> None:
        super().__init__(device)
        if model_name:
            self.revision = model_name
            self.name = model_name.split("/")[-1]

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ModuleNotFoundError as error:
                raise RepresentationError(
                    "transformers is not installed (pip install transformers)"
                ) from error
            tokenizer = AutoTokenizer.from_pretrained(self.revision)
            model = AutoModel.from_pretrained(self.revision).eval().to(self.device)
            self._model = (tokenizer, model)
        return self._model

    def encode(self, unit: RepresentationUnit) -> Any:
        torch = self._torch()
        tokenizer, model = self._ensure()
        batch = tokenizer(unit.sequence, return_tensors="pt", add_special_tokens=True)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(**batch).last_hidden_state[0]
        residues = output[1 : 1 + len(unit.sequence)]
        if residues.shape[0] != len(unit.sequence):
            raise RepresentationError("ESM-2 returned an unexpected residue count")
        if residues.shape[1] != self.dim:
            raise RepresentationError(
                f"ESM-2 width {residues.shape[1]} does not match the declared {self.dim}"
            )
        return residues.to(torch.float32).cpu()


class EsmIf1Backend(_TorchBackend):
    """ESM-IF1 inverse-folding encoder latents for one chain's own backbone."""

    name = "esm_if1_gvp4_t16_142M_UR50"
    dim = 512
    revision = "esm_if1_gvp4_t16_142M_UR50"
    requires_backbone = True

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                import esm  # noqa: F401
                import esm.inverse_folding  # noqa: F401
            except ModuleNotFoundError as error:
                raise RepresentationError(
                    "fair-esm is not installed (pip install fair-esm)"
                ) from error
            import esm as esm_module

            model, alphabet = esm_module.pretrained.esm_if1_gvp4_t16_142M_UR50()
            model = model.eval().to(self.device)
            self._model = (model, alphabet)
        return self._model

    def encode(self, unit: RepresentationUnit) -> Any:
        torch = self._torch()
        import esm.inverse_folding as inverse_folding

        model, alphabet = self._ensure()
        coordinates = unit.backbone
        if coordinates is None:
            raise RepresentationError("ESM-IF1 requires a backbone")
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 3):
            raise RepresentationError("backbone must have shape [residue, 3, 3] for N, CA, C")
        if coordinates.shape[0] != len(unit.sequence):
            raise RepresentationError(
                f"backbone has {coordinates.shape[0]} residues for a "
                f"{len(unit.sequence)}-residue sequence"
            )
        # fair-esm's get_encoder_output builds its batch with no device, so with
        # the model on CUDA it raises "expected all tensors to be on the same
        # device". CoordBatchConverter accepts a device; the wrapper simply never
        # passes one, so do the conversion here instead of falling back to CPU.
        converter = inverse_folding.util.CoordBatchConverter(alphabet)
        with torch.no_grad():
            batch_coordinates, confidence, _, _, padding_mask = converter(
                [(coordinates.numpy(), None, None)], device=self.device
            )
            encoder_out = model.encoder.forward(
                batch_coordinates, padding_mask, confidence, return_all_hiddens=False
            )
            # CoordBatchConverter pads one row at each end, so the encoder
            # returns [length + 2, batch, dim]; drop both to recover the residues.
            representation = encoder_out["encoder_out"][0][1:-1, 0]
        if representation.shape[0] != len(unit.sequence):
            raise RepresentationError("ESM-IF1 returned an unexpected residue count")
        return representation.to(torch.float32).cpu()


BACKENDS = {
    "antiberty": AntiBertyBackend,
    "esm2": Esm2Backend,
    "esm_if1": EsmIf1Backend,
}


def build_backend(name: str, device: str = "cpu") -> Any:
    """Construct a backend by name without importing the others."""
    if name not in BACKENDS:
        raise RepresentationError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](device=device)
