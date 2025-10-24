import torch
from typing import Optional, List, Tuple


class Proposer:
    """
    Simple proposer ψ for de novo ligand initialization.
    - Propose initial ligand atom positions around a pocket center (Gaussian/ball).
    - Optionally propose element classes from a prior.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    @staticmethod
    def _estimate_pocket_center(protein_pos: torch.Tensor) -> torch.Tensor:
        if protein_pos is None or protein_pos.numel() == 0:
            return torch.zeros(3, device=protein_pos.device if isinstance(protein_pos, torch.Tensor) else 'cpu')
        return protein_pos.mean(dim=0)

    def propose(self, num_atoms: int, center: torch.Tensor, sigma: float = 4.0,
                element_prior: Optional[List[int]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        center = center.to(self.device)
        pos = center.unsqueeze(0) + sigma * torch.randn(num_atoms, 3, device=self.device)
        elem = None
        if element_prior is not None and len(element_prior) > 0:
            # sample elements from prior class ids
            probs = torch.tensor(element_prior, dtype=torch.float, device=self.device)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, num_samples=num_atoms, replacement=True)
            elem = idx
        return pos, elem

    def propose_from_receptor(self, protein_pos: torch.Tensor, num_atoms: int,
                               sigma: float = 1.0, element_prior: Optional[List[int]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        center = self._estimate_pocket_center(protein_pos)
        return self.propose(num_atoms=num_atoms, center=center, sigma=sigma, element_prior=element_prior)


