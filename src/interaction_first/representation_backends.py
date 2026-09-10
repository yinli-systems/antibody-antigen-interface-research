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

from pathlib import Path

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




def place_backbone_oxygen(coordinates: Any) -> Any:
    """Add the carbonyl oxygen to an N, CA, C backbone by ideal geometry.

    ProteinMPNN featurises N, CA, C and O. The backbone store here holds only the
    first three, because ESM-IF1 needs no more, so O is constructed rather than
    read. It sits in the peptide plane at 1.231 A from C, along the direction
    that bisects the two bonds leaving C -- away from CA and away from the next
    residue's N:

        d = normalise( normalise(C - CA) + normalise(C - N_next) )
        O = C + 1.231 * d

    The final residue has no next N. Its O is placed along ``C - CA`` alone,
    which is the same construction with the missing term dropped.

    This is an idealisation, and it is the same idealisation for every arm --
    aligned, shuffled and control all receive it -- so it cannot produce a
    difference between them. It is recorded because a reader comparing this
    encoder to ESM-IF1 should know that one of them was given a reconstructed
    atom and the other was not.
    """
    import torch

    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    nitrogen, alpha, carbon = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    def unit(vector):
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    away_from_alpha = unit(carbon - alpha)
    direction = away_from_alpha.clone()
    if coordinates.shape[0] > 1:
        away_from_next = unit(carbon[:-1] - nitrogen[1:])
        direction[:-1] = unit(away_from_alpha[:-1] + away_from_next)
    oxygen = carbon + 1.231 * direction
    return torch.stack([nitrogen, alpha, carbon, oxygen], dim=1)


class ProteinMpnnBackend(_TorchBackend):
    """ProteinMPNN encoder latents for one chain's own backbone.

    A second structure-conditioned per-residue encoder, chosen because it shares
    no architecture and no training objective with ESM-IF1: a message-passing
    network over a k-nearest-neighbour graph of backbone atoms, against a
    GVP-transformer. What it shares is the input -- the same backbone store, the
    same chains, the same manifest -- which is what makes the comparison a
    comparison.
    """

    name = "proteinmpnn_v_48_020"
    dim = 128
    revision = "proteinmpnn_v_48_020"
    requires_backbone = True
    # Set by the caller to the checkout that holds protein_mpnn_utils.py and the
    # vanilla weights; there is no packaged distribution whose provenance this
    # project is willing to assume.
    checkout: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            import sys

            if self.checkout is None:
                raise RepresentationError(
                    "ProteinMpnnBackend.checkout must point at a ProteinMPNN checkout"
                )
            checkout = str(self.checkout)
            if checkout not in sys.path:
                sys.path.insert(0, checkout)
            try:
                from protein_mpnn_utils import ProteinMPNN
            except ModuleNotFoundError as error:
                raise RepresentationError(
                    f"protein_mpnn_utils.py is not under {checkout}"
                ) from error
            import torch

            weights = Path(checkout) / "vanilla_model_weights" / "v_48_020.pt"
            if not weights.is_file():
                raise RepresentationError(f"no weights at {weights}")
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            model = ProteinMPNN(
                num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                num_encoder_layers=3, num_decoder_layers=3,
                augment_eps=0.0, k_neighbors=int(checkpoint["num_edges"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            self._model = model.eval().to(self.device)
        return self._model

    def encode(self, unit: RepresentationUnit) -> Any:
        torch = self._torch()

        import sys

        if str(self.checkout) not in sys.path:
            sys.path.insert(0, str(self.checkout))
        from protein_mpnn_utils import gather_nodes

        model = self._ensure()
        if unit.backbone is None:
            raise RepresentationError("ProteinMPNN requires a backbone")
        coordinates = torch.as_tensor(unit.backbone, dtype=torch.float32)
        if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 3):
            raise RepresentationError("backbone must have shape [residue, 3, 3] for N, CA, C")
        if coordinates.shape[0] != len(unit.sequence):
            raise RepresentationError(
                f"backbone has {coordinates.shape[0]} residues for a "
                f"{len(unit.sequence)}-residue sequence"
            )
        length = coordinates.shape[0]
        with torch.no_grad():
            atoms = place_backbone_oxygen(coordinates).unsqueeze(0).to(self.device)
            mask = torch.ones(1, length, dtype=torch.float32, device=self.device)
            residue_index = torch.arange(length, device=self.device).unsqueeze(0)
            chain_encoding = torch.ones(1, length, dtype=torch.long, device=self.device)

            # The encoder half of ProteinMPNN.forward, which is where the
            # structure-conditioned per-residue representation lives. The decoder
            # is autoregressive over sequence and is not run: nothing here is
            # conditioned on the sequence, exactly as ESM-IF1's encoder output is
            # not.
            edges, edge_index = model.features(atoms, mask, residue_index, chain_encoding)
            nodes = torch.zeros(
                (edges.shape[0], edges.shape[1], edges.shape[-1]), device=edges.device
            )
            hidden = model.W_e(edges)
            attend = gather_nodes(mask.unsqueeze(-1), edge_index).squeeze(-1)
            attend = mask.unsqueeze(-1) * attend
            for layer in model.encoder_layers:
                nodes, hidden = layer(nodes, hidden, edge_index, mask, attend)
            representation = nodes[0]
        if representation.shape[0] != len(unit.sequence):
            raise RepresentationError("ProteinMPNN returned an unexpected residue count")
        return representation.to(torch.float32).cpu()


BACKENDS = {
    "antiberty": AntiBertyBackend,
    "esm2": Esm2Backend,
    "esm_if1": EsmIf1Backend,
    "proteinmpnn": ProteinMpnnBackend,
}


def build_backend(name: str, device: str = "cpu") -> Any:
    """Construct a backend by name without importing the others."""
    if name not in BACKENDS:
        raise RepresentationError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](device=device)
