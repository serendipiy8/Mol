import inspect
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torch_geometric.data import Data

from .losses import DiffusionLoss
from .diffusion_process import SoftMaskDiffusionProcess


class DiffusionTrainer:
    def __init__(self, diffusion_process: SoftMaskDiffusionProcess, loss_fn: DiffusionLoss,
                 optimizer: torch.optim.Optimizer, device: str = 'cuda',lambda_feat: float = 1.0, lambda_coord: float = 1.0,
                 grad_clip_norm: float = 1.0,aggregate_all_t: bool = False,
                 protein_encoder: Optional[torch.nn.Module] = None,
                 ligand_encoder: Optional[torch.nn.Module] = None,
                 encoder_edge_radius: float = 5.0,
                 lambda_eps: float = 0.0,
                 lambda_tau_smooth: float = 0.0,
                 lambda_tau_rank: float = 0.0,
                 lambda_atom_type: float = 0.0,
                 lambda_bond: float = 0.1,
                 debug_atom_type: bool = False,
                 normalize_coord_loss: bool = False, coord_use_kabsch: bool = True, coord_debug: bool = False, tau_as_t: bool = False, kappa: float = 5.0):

        self.diffusion_process = diffusion_process
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.lambda_feat = lambda_feat
        self.grad_clip_norm = grad_clip_norm
        self.lambda_coord = float(lambda_coord)
        self.aggregate_all_t = aggregate_all_t

        self.protein_encoder = protein_encoder
        self.ligand_encoder = ligand_encoder
        self.encoder_edge_radius = float(encoder_edge_radius)
        self.lambda_eps = float(lambda_eps)
        self.lambda_tau_smooth = float(lambda_tau_smooth)
        self.lambda_tau_rank = float(lambda_tau_rank)
        self.lambda_atom_type = float(lambda_atom_type)
        self.lambda_bond = float(lambda_bond)
        self.debug_atom_type = bool(debug_atom_type)
        self.normalize_coord_loss = bool(normalize_coord_loss)
        self._dbg_counter = 0
        self.coord_use_kabsch = bool(coord_use_kabsch)
        self.coord_debug = bool(coord_debug)
        self.tau_as_t = bool(tau_as_t)
        self.kappa = float(kappa)

    @staticmethod
    def _infer_element_labels_from_feat(feat: torch.Tensor) -> Optional[torch.Tensor]:
        """Infer atomic numbers by scanning feature columns and selecting the most plausible Z column.
        Returns long tensor or None if unreliable.
        """
        try:
            if not isinstance(feat, torch.Tensor) or feat.ndim != 2 or feat.size(1) < 1:
                return None
            palette = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35, 53], dtype=torch.long)
            best_ratio = 0.0
            best_col = None
            for j in range(int(feat.size(1))):
                col = feat[:, j].detach().cpu().to(torch.float32)
                col_int = torch.round(col).to(torch.long)
                if (col - col_int.to(col.dtype)).abs().mean().item() > 0.05:
                    continue
                if col_int.numel() == 0:
                    continue
                vmin = int(col_int.min().item())
                vmax = int(col_int.max().item())
                uniq = torch.unique(col_int)
                if vmin < 1 or vmax > 53 or uniq.numel() < 3:
                    continue
                isin = (col_int.unsqueeze(-1) == palette.unsqueeze(0)).any(dim=-1)
                ratio = float(isin.float().mean().item())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_col = col_int
            if best_col is not None and best_ratio >= 0.6:
                return best_col
            return None
        except Exception:
            return None

    def _build_radius_edges(self, pos: torch.Tensor, radius: float) -> torch.Tensor:
        """Naive radius graph (cpu/gpu), returns edge_index [2, E]."""
        if pos.ndim != 2 or pos.size(-1) != 3:
            return None
        with torch.no_grad():
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist2 = torch.sum(diff * diff, dim=-1)
            mask = (dist2 <= (radius * radius)) & (~torch.eye(pos.size(0), device=pos.device, dtype=torch.bool))
            src, dst = mask.nonzero(as_tuple=True)
            if src.numel() == 0:
                return None
            edge_index = torch.stack([src, dst], dim=0)
            return edge_index

    def _encode_context(self, batch: Data) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute g_P and h_lig via optional encoders; return (g_P, h_lig)."""
        g_P = None
        h_lig = None
        if self.protein_encoder is None and self.ligand_encoder is None:
            return g_P, h_lig

        protein_pos = self._get_field(batch, 'protein_pos')
        protein_feat = self._get_field(batch, 'protein_atom_feature')
        ligand_pos = self._get_field(batch, 'ligand_pos')
        ligand_feat = self._get_field(batch, 'ligand_atom_feature')

        if isinstance(protein_pos, torch.Tensor):
            protein_pos = protein_pos.to(self.device)
        if isinstance(protein_feat, torch.Tensor):
            protein_feat = protein_feat.to(self.device)
        if isinstance(ligand_pos, torch.Tensor):
            ligand_pos = ligand_pos.to(self.device)
        if isinstance(ligand_feat, torch.Tensor):
            ligand_feat = ligand_feat.to(self.device)

        # Build simple radius edges for encoders if not provided
        pp_edge_index = self._get_field(batch, 'pp_edge_index')
        if pp_edge_index is None and isinstance(protein_pos, torch.Tensor):
            pp_edge_index = self._build_radius_edges(protein_pos, self.encoder_edge_radius)

        ll_edge_index = self._get_field(batch, 'll_edge_index')
        if ll_edge_index is None and isinstance(ligand_pos, torch.Tensor):
            # small radius for ligand local structure; reuse encoder_edge_radius/2
            ll_edge_index = self._build_radius_edges(ligand_pos, max(2.0, self.encoder_edge_radius * 0.5))

        # Run encoders without gradient by default (feature providers); can be made trainable by user
        with torch.no_grad():
            if self.protein_encoder is not None and isinstance(protein_feat, torch.Tensor) and isinstance(protein_pos, torch.Tensor):
                try:
                    prot_out = self.protein_encoder(node_feat=protein_feat, pos=protein_pos, edge_index=pp_edge_index)
                    if isinstance(prot_out, (tuple, list)) and len(prot_out) >= 2:
                        if len(prot_out) == 3:
                            _, g_P_val, g_vec_val = prot_out
                        else:
                            _, g_P_val = prot_out[0], prot_out[1]
                            g_vec_val = None
                    elif isinstance(prot_out, dict) and 'g_P' in prot_out:
                        g_P_val = prot_out['g_P']
                        g_vec_val = prot_out.get('g_vec', None)
                    else:
                        g_P_val = protein_feat.mean(dim=0, keepdim=False)
                        g_vec_val = None
                    if isinstance(g_P_val, torch.Tensor):
                        g_P = g_P_val.to(self.device)
                    if isinstance(g_vec_val, torch.Tensor) and g_vec_val.numel() == 3:
                        try:
                            setattr(batch, 'g_P_vec', g_vec_val.to(self.device))
                        except Exception:
                            pass
                except Exception:
                    pass

            if self.ligand_encoder is not None and isinstance(ligand_feat, torch.Tensor) and isinstance(ligand_pos, torch.Tensor):
                try:
                    lig_out = self.ligand_encoder(node_feat=ligand_feat, pos=ligand_pos, edge_index=ll_edge_index)
                    if isinstance(lig_out, (tuple, list)):
                        h_lig_val = lig_out[0]
                    elif isinstance(lig_out, dict) and 'h' in lig_out:
                        h_lig_val = lig_out['h']
                    else:
                        h_lig_val = lig_out
                    if isinstance(h_lig_val, torch.Tensor):
                        h_lig = h_lig_val.to(self.device)
                except Exception:
                    pass

        if isinstance(g_P, torch.Tensor):
            try:
                setattr(batch, 'g_P', g_P)
            except Exception:
                pass
        if isinstance(h_lig, torch.Tensor):
            try:
                setattr(batch, 'h_t_override', h_lig)
            except Exception:
                pass

        return g_P, h_lig

    def _unwrap_batch(self, batch):
        if isinstance(batch, dict):
            for k in ('data', 'batch'):
                if k in batch and batch[k] is not None:
                    return batch[k]
            for v in batch.values():
                if v is not None:
                    return v
            return batch
        if isinstance(batch, (list, tuple)):
            for it in batch:
                if it is not None:
                    return it
        return batch

    def _assert_batch(self, batch: Data) -> None:
        ligand_pos = self._get_field(batch, 'ligand_pos')
        if ligand_pos is None:
            raise ValueError('Batch must contain ligand_pos tensor')

    def _sample_tau_logistic_normal(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        # Logistic-Normal reparam: tau = sigmoid(mu + exp(clamp(log_sigma))*eps)
        eps = torch.randn_like(mu)
        sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=10.0))
        u = mu + sigma * eps
        return torch.sigmoid(u)

    def _get_tau_params_from_model(self, model, batch: Data):
        # Prefer explicit tau param head if available
        if hasattr(model, 'predict_tau_params'):
            fn = getattr(model, 'predict_tau_params')
        elif hasattr(model, 'q_phi_head'):
            fn = getattr(model, 'q_phi_head')
        else:
            return None

        # Ensure critical fields are on the same device as the model before calling
        try:
            model_device = next(model.parameters()).device
            lig_feat = self._get_field(batch, 'ligand_atom_feature')
            if isinstance(lig_feat, torch.Tensor) and lig_feat.device != model_device:
                try:
                    setattr(batch, 'ligand_atom_feature', lig_feat.to(model_device))
                except Exception:
                    pass

            try:
                in_req = getattr(model, 'qphi_feat_proj').in_features
                lig_feat = self._get_field(batch, 'ligand_atom_feature')
                if isinstance(lig_feat, torch.Tensor):
                    lf = lig_feat.to(model_device)
                    if lf.dim() == 1:
                        lf = lf.view(1, -1)
                    if lf.size(-1) != in_req:
                        if lf.size(-1) < in_req:
                            pad = torch.zeros(lf.size(0), in_req - lf.size(-1), device=lf.device, dtype=lf.dtype)
                            lf = torch.cat([lf, pad], dim=-1)
                        else:
                            lf = lf[:, :in_req]
                        setattr(batch, 'ligand_atom_feature', lf)
            except Exception:
                pass
            g_P = getattr(batch, 'g_P', None)
            if isinstance(g_P, torch.Tensor) and g_P.device != model_device:
                try:
                    setattr(batch, 'g_P', g_P.to(model_device))
                except Exception:
                    pass
        except Exception:
            pass

        sig = inspect.signature(fn)
        if 'batch' in sig.parameters:
            out = fn(batch)
        else:
            return None

        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, log_sigma = out
            return mu, log_sigma
        return None

    def _assert_shapes_pre(self, x0: torch.Tensor, tau: torch.Tensor, t: torch.Tensor) -> None:
        if x0.ndim != 2 or x0.size(-1) != 3:
            raise ValueError(f'ligand_pos must be [N,3], got {tuple(x0.shape)}')
        if tau.ndim != 1 or tau.size(0) != x0.size(0):
            raise ValueError(f'tau must be [N], got {tuple(tau.shape)} vs N={x0.size(0)}')
        if t.ndim != 1 or t.size(0) != x0.size(0):
            raise ValueError(f't must be [N], got {tuple(t.shape)} vs N={x0.size(0)}')
        dev = self.device
        for ten in (x0, tau, t):
            if ten.device.type != torch.device(dev).type:
                raise ValueError(f'Device mismatch: expected {dev}, got {ten.device}')

    def _assert_shapes_post_coord(self, x_t: torch.Tensor, s_t: torch.Tensor, x0: torch.Tensor) -> None:
        if x_t.shape != x0.shape:
            raise ValueError(f'x_t shape {tuple(x_t.shape)} must match x0 {tuple(x0.shape)}')
        if s_t.ndim != 1 or s_t.size(0) != x0.size(0):
            raise ValueError(f's_t must be [N], got {tuple(s_t.shape)} vs N={x0.size(0)}')
        for ten in (x_t, s_t):
            if ten.device != x0.device:
                raise ValueError('Post-forward device mismatch on coordinates path')

    def _assert_shapes_post_multi(self, h0: torch.Tensor, h_t: torch.Tensor, s_t: torch.Tensor, x0: torch.Tensor) -> None:
        if h0.ndim != 2 or h_t.ndim != 2 or h0.shape != h_t.shape:
            raise ValueError(f'h0/h_t must be [N,d] same shape, got {tuple(h0.shape)} vs {tuple(h_t.shape)}')
        if h0.size(0) != x0.size(0):
            raise ValueError('Feature N must equal coordinate N')
        if s_t.ndim != 1 or s_t.size(0) != x0.size(0):
            raise ValueError(f's_t must be [N], got {tuple(s_t.shape)}')
        for ten in (h0, h_t, s_t):
            if ten.device != x0.device:
                raise ValueError('Post-forward device mismatch on multi-modal path')

    def _get_field(self, obj, name):
        # Robust getter for dict or PyG Batch/Data
        if isinstance(obj, dict):
            return obj.get(name)
        try:
            if name in obj:
                return getattr(obj, name)
        except Exception:
            pass
        try:
            return getattr(obj, name)
        except Exception:
            return None

    def _ensure_fields(self, batch: Data) -> None:
        """Soft-ensure canonical fields exist on batch by mapping known aliases."""
        # ligand_pos fallbacks
        if self._get_field(batch, 'ligand_pos') is None:
            alt = self._get_field(batch, 'ligand_context_pos')
            if isinstance(alt, torch.Tensor):
                try:
                    setattr(batch, 'ligand_pos', alt)
                except Exception:
                    pass
        # ligand_atom_feature fallbacks
        if self._get_field(batch, 'ligand_atom_feature') is None:
            alt = self._get_field(batch, 'ligand_context_feature')
            if isinstance(alt, torch.Tensor):
                try:
                    setattr(batch, 'ligand_atom_feature', alt)
                except Exception:
                    pass

    def _unwrap_batch(self, batch):
        if isinstance(batch, dict):
            # common keys
            for k in ('data', 'batch'):
                if k in batch and batch[k] is not None:
                    cand = batch[k]
                    if self._get_field(cand, 'ligand_pos') is not None:
                        return cand
            # search values for a Data/Batch-like payload
            for v in batch.values():
                if v is None:
                    continue
                if self._get_field(v, 'ligand_pos') is not None:
                    return v
                try:
                    if hasattr(v, 'keys') and len(list(v.keys)) > 0:
                        return v
                except Exception:
                    pass
            # fallback: return the dict itself
            return batch
        if isinstance(batch, (list, tuple)):
            for item in batch:
                if item is None:
                    continue
                # candidate if has ligand_pos or behaves like Data/Batch
                if self._get_field(item, 'ligand_pos') is not None:
                    return item
                try:
                    if hasattr(item, 'keys') and len(list(item.keys)) > 0:
                        return item
                except Exception:
                    pass
            for item in batch:
                if item is not None:
                    return item
        return batch

    def _assert_model_outputs(self, x0_pred: torch.Tensor, x0: torch.Tensor, h0_pred: torch.Tensor = None, h0: torch.Tensor = None) -> None:
        if x0_pred.shape != x0.shape:
            raise ValueError(f'x0_pred shape {tuple(x0_pred.shape)} must match x0 {tuple(x0.shape)}')
        if x0_pred.device != x0.device:
            raise ValueError('x0_pred device must match x0 device')
        if h0_pred is not None:
            if h0 is None:
                raise ValueError('h0_pred provided but h0 is None')
            if h0_pred.shape != h0.shape:
                raise ValueError(f'h0_pred shape {tuple(h0_pred.shape)} must match h0 {tuple(h0.shape)}')
            if h0_pred.device != h0.device:
                raise ValueError('h0_pred device must match h0 device')

    def training_step(self, batch: Data, model, soft_mask_transform) -> Dict[str, float]:

        batch = self._unwrap_batch(batch)
        self._ensure_fields(batch)
        self._assert_batch(batch)

        x0_field = self._get_field(batch, 'ligand_pos')
        if x0_field is None:
            raise ValueError('Batch must contain ligand_pos tensor')
        
        _center = x0_field.mean(dim=0, keepdim=True)
        x0_field = x0_field - _center
        x0 = x0_field.to(self.device)


        # Tau参数
        tau_mu = None
        tau_log_sigma = None
        tau_params = self._get_tau_params_from_model(model, batch)
        if tau_params is not None:
            tau_mu, tau_log_sigma = tau_params
            tau_mu = tau_mu.to(self.device)
            tau_log_sigma = tau_log_sigma.to(self.device)
            tau = self._sample_tau_logistic_normal(tau_mu, tau_log_sigma)
            batch.tau_mu = tau_mu
            batch.tau_log_sigma = tau_log_sigma
        else:
            if hasattr(batch, 'tau') and batch.tau is not None:
                tau = batch.tau.to(self.device)
            else:
                tau = torch.full((x0.size(0),), 0.5, device=self.device)
        batch.tau = tau

        # 编码上下文
        self._encode_context(batch)
        has_feat = self._get_field(batch, 'ligand_atom_feature') is not None

        # 采样时间步
        if not self.aggregate_all_t:
            t_scalar = torch.randint(0, self.diffusion_process.num_steps, (1,), device=self.device, dtype=torch.long).item()
            t = torch.full((x0.size(0),), t_scalar, device=self.device, dtype=torch.long)
            self._assert_shapes_pre(x0, tau, t)

        # 模型调用封装
        def _call_model(model, x_t_tensor, s_t_tensor, t_tensor, batch_ctx=None, h_t_tensor=None):
            forward_fn = getattr(model, 'forward', model)
            sig = inspect.signature(forward_fn).parameters
            args = [x_t_tensor, s_t_tensor, t_tensor]
            if 'batch' in sig and batch_ctx is not None:
                args.append(batch_ctx)
            if 'h_t' in sig and h_t_tensor is not None:
                if 'batch' in sig:
                    args.append(h_t_tensor)
                else:
                    args.append(h_t_tensor)
            return forward_fn(*args)

        t_scalar_log = None
        s_stats = {}

        if not self.aggregate_all_t:
            if has_feat:
                h0_field = getattr(batch, 'h_t_override', None)
                if isinstance(h0_field, torch.Tensor):
                    h0 = h0_field.to(self.device)
                else:
                    h0_field = self._get_field(batch, 'ligand_atom_feature')
                    h0 = h0_field.to(self.device)

                _old_prot = getattr(batch, 'protein_pos', None)
                _old_lig = getattr(batch, 'ligand_pos', None)
                # 用同一个中心（配体均值）对蛋白/配体一起中心化，保持相对几何
                try:
                    if isinstance(_old_prot, torch.Tensor):
                        batch.protein_pos = _old_prot - _center.to(_old_prot.device, _old_prot.dtype)
                    if isinstance(_old_lig, torch.Tensor) and _old_lig.shape == x0.shape:
                        batch.ligand_pos = _old_lig - _center.to(_old_lig.device, _old_lig.dtype)
                except Exception:
                    pass

                x_t, h_t, s_t = self.diffusion_process.forward_process_multi_modal(x0, h0, tau, t, soft_mask_transform)
                self._assert_shapes_post_coord(x_t, s_t, x0)
                self._assert_shapes_post_multi(h0, h_t, s_t, x0)

                # 强制 s_t 重新计算
                try:
                    alpha = (t.to(torch.float32) / float(max(1, self.diffusion_process.num_steps - 1)))
                    sigma_t = self.diffusion_process.get_sigma_t(t)
                    s_new = torch.sigmoid(self.kappa * (tau.to(alpha.device, alpha.dtype) - alpha)).to(s_t.dtype)
                    denom = (1.0 - s_t).unsqueeze(-1) * sigma_t.unsqueeze(-1) + 1e-8
                    eps_c = (x_t - s_t.unsqueeze(-1) * x0) / denom
                    x_t = s_new.unsqueeze(-1) * x0 + (1.0 - s_new).unsqueeze(-1) * sigma_t.unsqueeze(-1) * eps_c
                    s_t = s_new
                except Exception:
                    pass

                model_out = _call_model(model, x_t, s_t, t, batch_ctx=batch, h_t_tensor=h_t)
                if isinstance(model_out, (tuple, list)) and len(model_out) == 2:
                    x0_pred, h0_pred = model_out
                else:
                    x0_pred, h0_pred = model_out, None
                self._assert_model_outputs(x0_pred, x0, h0_pred, h0)

            else:
                _old_prot = getattr(batch, 'protein_pos', None)
                _old_lig = getattr(batch, 'ligand_pos', None)
                try:
                    if isinstance(_old_prot, torch.Tensor):
                        batch.protein_pos = _old_prot - _center.to(_old_prot.device, _old_prot.dtype)
                    if isinstance(_old_lig, torch.Tensor) and _old_lig.shape == x0.shape:
                        batch.ligand_pos = _old_lig - _center.to(_old_lig.device, _old_lig.dtype)
                except Exception:
                    pass

                x_t, s_t = self.diffusion_process.forward_process(x0, tau, t, soft_mask_transform)
                self._assert_shapes_post_coord(x_t, s_t, x0)

                try:
                    alpha = (t.to(torch.float32) / float(max(1, self.diffusion_process.num_steps - 1)))
                    sigma_t = self.diffusion_process.get_sigma_t(t)
                    s_new = torch.sigmoid(self.kappa * (tau.to(alpha.device, alpha.dtype) - alpha)).to(s_t.dtype)
                    denom = (1.0 - s_t).unsqueeze(-1) * sigma_t.unsqueeze(-1) + 1e-8
                    eps_c = (x_t - s_t.unsqueeze(-1) * x0) / denom
                    x_t = s_new.unsqueeze(-1) * x0 + (1.0 - s_new).unsqueeze(-1) * sigma_t.unsqueeze(-1) * eps_c
                    s_t = s_new
                except Exception:
                    pass

                if self.debug_atom_type:
                    try:
                        setattr(batch, '_dbg_forward', True)
                    except Exception:
                        pass

                x0_pred = _call_model(model, x_t, s_t, t, batch_ctx=batch)
                self._assert_model_outputs(x0_pred, x0)

            # 计算 loss
            sigma_coord = self.diffusion_process.sigmas[t]
            sigma_feat = self.diffusion_process.sigmas[t] if has_feat else sigma_coord

            def _maybe_norm(x_gt: torch.Tensor, x_pr: torch.Tensor):
                if self.normalize_coord_loss:
                    mu_g = x_gt.mean(dim=0, keepdim=True)
                    std_g = x_gt.std(dim=0, keepdim=True).clamp_min(1e-3)
                    return (x_gt - mu_g) / std_g, (x_pr - mu_g) / std_g
                return x_gt, x_pr

            x0_use, x0_pred_use = _maybe_norm(x0, x0_pred)
            if has_feat and h0_pred is not None:
                coord_loss_val = self.loss_fn.compute_loss(x0_use, x0_pred_use, s_t, sigma_coord)
                feat_loss_val = self.loss_fn.compute_loss(h0, h0_pred, s_t, sigma_feat)
                diffusion_loss = coord_loss_val + self.lambda_feat * feat_loss_val
            else:
                coord_loss_val = self.loss_fn.compute_loss(x0_use, x0_pred_use, s_t, sigma_coord)
                feat_loss_val = torch.tensor(0.0, device=self.device)
                diffusion_loss = coord_loss_val

            t_scalar_log = int(t[0].item()) if isinstance(t, torch.Tensor) and t.numel() > 0 else None
            s_stats = {'s_mean': float(s_t.mean().item()), 's_std': float(s_t.std(unbiased=False).item())}

        else:
            # aggregate_all_t 逻辑
            num_steps = int(self.diffusion_process.num_steps)
            agg_loss = torch.zeros((), device=self.device)
            agg_coord = torch.zeros((), device=self.device)
            agg_feat = torch.zeros((), device=self.device)
            steps_count = 0
            for t_val in range(num_steps):
                t_loop = torch.full((x0.size(0),), t_val, device=self.device, dtype=torch.long)
                self._assert_shapes_pre(x0, tau, t_loop)

                if has_feat:
                    h0_field = getattr(batch, 'h_t_override', None)
                    if isinstance(h0_field, torch.Tensor):
                        h0 = h0_field.to(self.device)
                    else:
                        h0_field = self._get_field(batch, 'ligand_atom_feature')
                        h0 = h0_field.to(self.device)

                    x_t, h_t, s_t = self.diffusion_process.forward_process_multi_modal(x0, h0, tau, t_loop, soft_mask_transform)
                    self._assert_shapes_post_coord(x_t, s_t, x0)
                    self._assert_shapes_post_multi(h0, h_t, s_t, x0)

                    model_out = _call_model(model, x_t, s_t, t_loop, batch_ctx=batch, h_t_tensor=h_t)
                    if isinstance(model_out, (tuple, list)) and len(model_out) == 2:
                        x0_pred, h0_pred = model_out
                    else:
                        x0_pred, h0_pred = model_out, None
                    self._assert_model_outputs(x0_pred, x0, h0_pred, h0)

                    sigma_coord = self.diffusion_process.sigmas[t_loop]
                    sigma_feat = self.diffusion_process.sigmas[t_loop]

                    x0_use, x0_pred_use = _maybe_norm(x0, x0_pred)
                    if h0_pred is not None:
                        coord_step = self.loss_fn.compute_loss(x0_use, x0_pred_use, s_t, sigma_coord)
                        feat_step = self.loss_fn.compute_loss(h0, h0_pred, s_t, sigma_feat)
                        step_loss = coord_step + self.lambda_feat * feat_step
                        agg_coord = agg_coord + coord_step
                        agg_feat = agg_feat + feat_step
                    else:
                        coord_step = self.loss_fn.compute_loss(x0_use, x0_pred_use, s_t, sigma_coord)
                        step_loss = coord_step
                        agg_coord = agg_coord + coord_step
                    steps_count += 1
                else:
                    x_t, s_t = self.diffusion_process.forward_process(x0, tau, t_loop, soft_mask_transform)
                    self._assert_shapes_post_coord(x_t, s_t, x0)
                    x0_pred = _call_model(model, x_t, s_t, t_loop, batch_ctx=batch)
                    self._assert_model_outputs(x0_pred, x0)
                    sigma_coord = self.diffusion_process.sigmas[t_loop]
                    x0_use, x0_pred_use = _maybe_norm(x0, x0_pred)
                    coord_step = self.loss_fn.compute_loss(x0_use, x0_pred_use, s_t, sigma_coord)
                    step_loss = coord_step
                    agg_coord = agg_coord + coord_step
                    steps_count += 1
                agg_loss = agg_loss + step_loss
            diffusion_loss = agg_loss / float(num_steps)
            coord_loss_val = agg_coord / float(max(1, steps_count))
            feat_loss_val = agg_feat / float(max(1, steps_count))
            t_scalar_log = -1
            s_stats = {}

        # KL loss for τ parameters: prefer model-provided; else fallback to batch fields
        if tau_mu is not None and tau_log_sigma is not None:
            kl_loss = self.loss_fn.compute_kl_loss(tau_mu, tau_log_sigma)
        elif hasattr(batch, 'tau_mu') and hasattr(batch, 'tau_log_sigma') and batch.tau_mu is not None and batch.tau_log_sigma is not None:
            kl_loss = self.loss_fn.compute_kl_loss(batch.tau_mu.to(self.device), batch.tau_log_sigma.to(self.device))
        else:
            kl_loss = torch.tensor(0.0, device=self.device)

        # epsilon supervision removed

        # Optional bond supervision if available on batch
        bond_loss = torch.tensor(0.0, device=self.device)
        pred_logits = getattr(batch, 'pred_bond_logits', None)
        pred_index = getattr(batch, 'pred_bond_index', None)
        gt_bond = self._get_field(batch, 'ligand_bond_index')
        gt_bond_type = self._get_field(batch, 'ligand_bond_type')
        if isinstance(pred_logits, torch.Tensor) and isinstance(pred_index, torch.Tensor) and isinstance(gt_bond, torch.Tensor):
            bond_loss = self.loss_fn.compute_bond_loss(pred_logits.to(self.device),
                                                    pred_index.to(self.device),
                                                    gt_bond.to(self.device),
                                                    gt_bond_type.to(self.device) if isinstance(gt_bond_type, torch.Tensor) else None)

        # Tau smooth & rank regularization
        tau_reg = torch.tensor(0.0, device=self.device)
        if self.lambda_tau_smooth > 0.0:
            try:
                lig_pos = x0
                D = torch.cdist(lig_pos, lig_pos)
                mask = (D < 3.0) & (D > 0)  # small radius neighbors
                i, j = torch.nonzero(mask, as_tuple=True)
                if i.numel() > 0:
                    tau_i = tau[i]
                    tau_j = tau[j]
                    tau_reg = tau_reg + torch.mean((tau_i - tau_j) ** 2)
            except Exception:
                pass
        if self.lambda_tau_rank > 0.0:
            try:
                lig_pos = x0
                D = torch.cdist(lig_pos, lig_pos)
                N = D.size(0)
                if N > 1:
                    idx_i = torch.randint(0, N, (min(N*4, 1024),), device=self.device)
                    idx_j = torch.randint(0, N, (min(N*4, 1024),), device=self.device)
                    closer = (D[idx_i, idx_j] < D[idx_j, idx_i]).float()
                    margin = 0.0
                    tau_i = tau[idx_i]
                    tau_j = tau[idx_j]
                    rank_loss = torch.relu(margin - (tau_i - tau_j)) * closer + torch.relu(margin - (tau_j - tau_i)) * (1 - closer)
                    tau_reg = tau_reg + rank_loss.mean()
            except Exception:
                pass

        # Optional atom type classification loss
        atom_type_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_atom_type > 0.0:
            try:
                atom_logits = getattr(batch, 'pred_atom_type_logits', None)
                if isinstance(atom_logits, torch.Tensor):
                    lig_el = getattr(batch, 'ligand_element', None)
                    z = lig_el.to(self.device).to(torch.long) if isinstance(lig_el, torch.Tensor) else None
                    if z is None:
                        lig_feat0 = self._get_field(batch, 'ligand_atom_feature')
                        inferred = self._infer_element_labels_from_feat(lig_feat0) if isinstance(lig_feat0, torch.Tensor) else None
                        if isinstance(inferred, torch.Tensor):
                            z = inferred.to(self.device).to(torch.long)
                    if z is not None and atom_logits.size(0) == z.size(0):
                        palette = torch.tensor([1,6,7,8,9,15,16,17,35,53], device=self.device)
                        z_expand = z.unsqueeze(-1).to(torch.float32)
                        pal_expand = palette.unsqueeze(0).to(torch.float32)
                        dist = torch.abs(z_expand - pal_expand)
                        labels = torch.argmin(dist, dim=-1)
                        atom_type_loss = F.cross_entropy(atom_logits, labels)
            except Exception:
                atom_type_loss = torch.tensor(0.0, device=self.device)

        tau_smooth_loss = self.lambda_tau_smooth * tau_reg
        tau_rank_loss = self.lambda_tau_rank * tau_reg
        total_loss = (self.lambda_coord * diffusion_loss
                    + kl_loss
                    + self.lambda_bond * bond_loss
                        + tau_smooth_loss
                        + tau_rank_loss
                    + self.lambda_atom_type * atom_type_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.optimizer.param_groups[0]['params'] if p.requires_grad],
                max_norm=self.grad_clip_norm
            )
        # gradient norms (coord vs feat) before optimizer.step()
        grad_coord = torch.tensor(0.0, device=self.device)
        grad_feat = torch.tensor(0.0, device=self.device)
        try:
            # model is passed to training_step; compute by name grouping
            if model is not None:
                coord_sq = 0.0
                feat_sq = 0.0
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g2 = p.grad.detach().pow(2).sum().item()
                    if ('coord_layers' in name) or ('out_coord' in name) or ('coord_mlp' in name):
                        coord_sq += g2
                    elif ('feat_layers' in name) or ('out_feat' in name):
                        feat_sq += g2
                grad_coord = torch.tensor(coord_sq ** 0.5, device=self.device)
                grad_feat = torch.tensor(feat_sq ** 0.5, device=self.device)
        except Exception:
            pass
        self.optimizer.step()

        out_logs = {
            'total_loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'coord_loss': float(coord_loss_val.item()) if 'coord_loss_val' in locals() and isinstance(coord_loss_val, torch.Tensor) else 0.0,
            'feat_loss': float(feat_loss_val.item()) if 'feat_loss_val' in locals() and isinstance(feat_loss_val, torch.Tensor) else 0.0,
            'kl_loss': kl_loss.item(),
            'bond_loss': bond_loss.item() if isinstance(bond_loss, torch.Tensor) else 0.0,
            'atom_type_loss': atom_type_loss.item() if isinstance(atom_type_loss, torch.Tensor) else 0.0,
            'tau_smooth_loss': float(tau_smooth_loss.item()) if isinstance(tau_smooth_loss, torch.Tensor) else 0.0,
            'tau_rank_loss': float(tau_rank_loss.item()) if isinstance(tau_rank_loss, torch.Tensor) else 0.0,
            'grad_coord': float(grad_coord.item()) if isinstance(grad_coord, torch.Tensor) else 0.0,
            'grad_feat': float(grad_feat.item()) if isinstance(grad_feat, torch.Tensor) else 0.0,
        }
        if t_scalar_log is not None:
            out_logs['t'] = t_scalar_log
        if s_stats:
            out_logs.update(s_stats)

        return out_logs
