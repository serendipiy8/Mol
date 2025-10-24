import torch

from src.model.denoiser import EGNNDenoiser
from src.model.encoders import ProteinEncoder, LigandEncoder
from src.model.diffusion import SoftMaskDiffusionProcess, DiffusionLoss, DiffusionTrainer


def build_model_and_trainer(args, node_dim: int, device: torch.device):
    if args.protein_encoder_se3:
        try:
            from src.model.encoders.se3_encoder import SE3ProteinEncoder
            protein_encoder = SE3ProteinEncoder(in_dim=node_dim, hidden_dim=node_dim).to(device)
        except Exception:
            protein_encoder = ProteinEncoder(in_dim=node_dim, hidden_dim=node_dim).to(device)
    else:
        protein_encoder = ProteinEncoder(in_dim=node_dim, hidden_dim=node_dim).to(device)
    ligand_encoder = LigandEncoder(in_dim=node_dim, hidden_dim=node_dim).to(device)

    model = EGNNDenoiser(
        node_dim=node_dim,
        edge_dim=1,
        hidden_dim=max(int(args.hidden_dim), node_dim),
        num_layers=int(args.num_layers),
        use_time=bool(getattr(args, 'use_time', True)),
        use_s_gate=bool(getattr(args, 'use_s_gate', True)),
        use_cross_mp=bool(args.use_cross_mp),
        cross_mp_hidden=None,
        cross_mp_bidirectional=False,
        cross_mp_use_distance=True,
        cross_mp_rbf_dim=0,
        cross_mp_use_direction=False,
        cross_mp_distance_sigma=4.0,
        cross_mp_dropout=0.0,
        cross_mp_every=int(getattr(args, 'cross_mp_every', 1)),
        use_protein_context=bool(args.use_protein_context),
        context_mode='film',
        context_dropout=float(args.context_dropout),
        use_dynamic_cross_edges=True,
        cross_topk=int(args.cross_topk),
        cross_radius=float(args.cross_radius),
        use_bond_head=bool(getattr(args, 'use_bond_head', True)),
        bond_hidden=64,
        bond_classes=int(getattr(args, 'bond_classes', 5)),
        bond_radius=2.2,
        use_atom_type_head=bool(getattr(args, 'use_atom_type_head', True)),
        atom_type_classes=int(getattr(args, 'atom_type_classes', 10)),
        use_dual_branch=bool(getattr(args, 'use_dual_branch', True)),
        coord_use_tanh=bool(getattr(args, 'coord_use_tanh', False)),
        coord_alpha_min=float(getattr(args, 'coord_alpha_min', 0.3)),
        debug_graph=bool(getattr(args, 'debug_graph', False)),
    ).to(device)

    diffusion = SoftMaskDiffusionProcess(
        num_steps=int(getattr(args, 'num_steps', 100)),
        sigma_min=float(getattr(args, 'sigma_min', 0.2)),
        sigma_max=float(getattr(args, 'sigma_max', 0.2)),
        sigma_schedule='linear',
        device=str(device)
    )
    # 提升早期梯度：切换为 MSE（后续可再改回 Huber）
    loss_fn = DiffusionLoss(loss_type='mse', reweighting=True, w_max=1e2)
    # Two-group optimizer: boost coordinate-related layers' LR
    base_lr = float(getattr(args, 'lr', 3e-4))
    coord_params = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('out_coord' in name) or ('coord_mlp' in name):
            coord_params.append(p)
        else:
            base_params.append(p)
    # revert coord head to base lr for stability; can tune later
    optimizer = torch.optim.Adam([
        { 'params': base_params, 'lr': base_lr },
        { 'params': coord_params, 'lr': base_lr * 4.0 },
    ])
    # If coord_debug is enabled, force single-step aggregation for detailed prints
    agg_all = bool(args.aggregate_all_t)
    if bool(getattr(args, 'coord_debug', False)):
        agg_all = False

    trainer = DiffusionTrainer(
        diffusion_process=diffusion,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=str(device),
        lambda_feat=float(getattr(args, 'lambda_feat', 10)),
        lambda_coord=float(getattr(args, 'lambda_coord', 0.5)),
        grad_clip_norm=1.0,
        aggregate_all_t=agg_all,
        protein_encoder=protein_encoder,
        ligand_encoder=ligand_encoder,
        encoder_edge_radius=5.0,
        lambda_eps=float(getattr(args, 'lambda_eps', 0.0)),
        lambda_tau_smooth=float(getattr(args, 'lambda_tau_smooth', 0.0)),
        lambda_tau_rank=float(getattr(args, 'lambda_tau_rank', 0.0)),
        lambda_atom_type=float(getattr(args, 'lambda_atom_type', 0.5)),
        lambda_bond=float(getattr(args, 'lambda_bond', 0.5)),
        debug_atom_type=bool(getattr(args, 'debug_atom_type', True)),
        normalize_coord_loss=bool(getattr(args, 'normalize_coord_loss', True)),
        coord_use_kabsch=bool(getattr(args, 'coord_use_kabsch', True)),
        coord_debug=bool(getattr(args, 'coord_debug', False)),
        tau_as_t=bool(getattr(args, 'tau_as_t', False)),
        kappa=float(getattr(args, 'kappa', 5.0)),
    )
    return model, trainer, diffusion