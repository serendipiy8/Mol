import math
import torch
import torch.nn.functional as F
from typing import Tuple


class DiffusionLoss:

    def __init__(self, loss_type: str = 'mse', reweighting: bool = False,
                 w_max: float = 1e3, eps: float = 1e-8):
        self.loss_type = loss_type
        self.reweighting = reweighting
        self.w_max = w_max
        self.eps = eps

    def compute_loss(self, x0: torch.Tensor, x0_pred: torch.Tensor,
                     s_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:

        if self.loss_type == 'mse':
            base_loss = F.mse_loss(x0_pred, x0, reduction='none')
        elif self.loss_type == 'l1':
            base_loss = F.l1_loss(x0_pred, x0, reduction='none')
        elif self.loss_type == 'huber':
            base_loss = F.huber_loss(x0_pred, x0, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        base_loss = base_loss.mean(dim=-1) 

        if self.reweighting:
            sigma_safe = torch.clamp(sigma_t, min=1e-4)
            s_safe = torch.clamp(1.0 - s_t, min=1e-4)
            weights = 1.0 / (s_safe ** 2 * sigma_safe ** 2 + self.eps)
            weights = torch.clamp(weights, max=self.w_max)
            weights = weights / (weights.mean() + self.eps)
            base_loss = base_loss * weights

        return base_loss.mean()

    def compute_loss_multi_modal(self, x0_coord: torch.Tensor, x0_pred: torch.Tensor,
                                 h0_feat: torch.Tensor, h0_pred: torch.Tensor,
                                 s_t: torch.Tensor, sigma_coord: torch.Tensor,
                                 sigma_feat: torch.Tensor, lambda_feat: float = 100) -> torch.Tensor:

        loss_coord = self.compute_loss(x0_coord, x0_pred, s_t, sigma_coord)
        loss_feat = self.compute_loss(h0_feat, h0_pred, s_t, sigma_feat)
        
        return loss_coord + lambda_feat * loss_feat

    def compute_kl_loss(self, mu: torch.Tensor, log_sigma: torch.Tensor,
                        prior_mu: float = 0.0, prior_sigma: float = 1.0) -> torch.Tensor:
        """
        KL(N(mu, sigma) || N(prior_mu, prior_sigma)) in closed form.
        Uses clamped log_sigma and stable log(prior_sigma) handling.
        """
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        sigma = torch.exp(log_sigma)

        prior_sigma_t = torch.tensor(prior_sigma, device=mu.device, dtype=mu.dtype)
        prior_mu_t = torch.tensor(prior_mu, device=mu.device, dtype=mu.dtype)
        log_prior_sigma = torch.log(prior_sigma_t)

        kl_div = 0.5 * (
            (sigma / prior_sigma_t) ** 2 +
            ((mu - prior_mu_t) / prior_sigma_t) ** 2 -
            1.0 +
            2.0 * (log_prior_sigma - log_sigma)
        )

        return kl_div.mean()

    def compute_bond_loss(self, pred_logits: torch.Tensor, pred_edge_index: torch.Tensor,
                          gt_edge_index: torch.Tensor = None, gt_bond_type: torch.Tensor = None,
                          reduction: str = 'mean') -> torch.Tensor:
        """Bond prediction loss on predicted edges only.

        - If gt_edge_index is None or pred has no edges, returns 0.
        - Builds labels by checking whether each predicted undirected edge exists in GT.
        - If gt_bond_type provided (multi-class), uses其类型；否则退化为二分类存在性。
        """
        if pred_logits is None or pred_logits.numel() == 0:
            return torch.tensor(0.0, device=pred_logits.device if isinstance(pred_logits, torch.Tensor) else 'cpu')
        if gt_edge_index is None or gt_edge_index.numel() == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        def _canonical_pairs(ei: torch.Tensor) -> torch.Tensor:
            a = ei[0].to(torch.long)
            b = ei[1].to(torch.long)
            lo = torch.minimum(a, b)
            hi = torch.maximum(a, b)
            return torch.stack([lo, hi], dim=0)

        pred_pairs = _canonical_pairs(pred_edge_index)
        gt_pairs = _canonical_pairs(gt_edge_index).t()  # [E_gt,2]

        # Hash pairs to ids for fast membership: id = lo*N + hi (approx; N inferred from max index+1)
        N = int(max(int(pred_pairs.max().item()) if pred_pairs.numel() > 0 else 0,
                    int(gt_pairs.max().item()) if gt_pairs.numel() > 0 else 0) + 1)
        pred_ids = pred_pairs[0] * N + pred_pairs[1]
        gt_ids = gt_pairs[:, 0] * N + gt_pairs[:, 1]

        # Labels: use type if available else 1/0 by membership
        # Use set membership via torch.isin (PyTorch>=1.10). Fallback: sort+search if needed.
        try:
            if gt_bond_type is not None and gt_bond_type.numel() > 0:
                # map gt edge id -> type
                unique_ids, inv = torch.unique(gt_ids, return_inverse=True)
                # build dict-like map by tensor ops
                # For simplicity, take first occurrence
                type_map = {}
                for k in range(unique_ids.size(0)):
                    type_map[int(unique_ids[k].item())] = int(gt_bond_type[k].item() if k < gt_bond_type.size(0) else 0)
                labels = torch.tensor([type_map.get(int(e.item()), 0) for e in pred_ids], device=pred_logits.device, dtype=torch.long)
            else:
                labels = torch.isin(pred_ids, gt_ids).to(torch.long)
        except Exception:
            gt_ids_sorted, _ = torch.sort(gt_ids)
            idx = torch.searchsorted(gt_ids_sorted, pred_ids)
            idx = torch.clamp(idx, max=max(gt_ids_sorted.numel() - 1, 0))
            exists = (gt_ids_sorted.numel() > 0) & (gt_ids_sorted[idx] == pred_ids)
            labels = exists.to(torch.long)

        # Expect pred_logits shape [E_pred, C]
        if pred_logits.dim() == 1:
            # If provided as logits for class-1 only, convert to two-class logits
            logits_pos = pred_logits
            logits = torch.stack([-logits_pos, logits_pos], dim=-1)
        else:
            logits = pred_logits

        loss = torch.nn.functional.cross_entropy(logits, labels.to(logits.device), reduction=reduction)
        return loss
