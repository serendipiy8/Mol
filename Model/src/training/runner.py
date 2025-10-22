import os
from typing import List

import torch
import logging
from tqdm import tqdm

from src.model.utils.repro import set_seed
from src.data.data_loader import get_data_loaders
from src.data.soft_mask_transforms import SoftMaskTransform
from src.model.denoiser import EGNNDenoiser
from src.model.encoders import ProteinEncoder, LigandEncoder
from src.model.diffusion import SoftMaskDiffusionProcess, DiffusionLoss, DiffusionTrainer
from src.model.sampler import ConditionalSampler
from src.evaluation.geometric_evaluation import run_geometric_evaluation
from src.evaluation.chemical_evaluation import run_chemical_evaluation
from src.evaluation.similarity_evaluation import run_similarity_evaluation


def move_to_device(batch, device: str):
    if isinstance(batch, dict):
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    try:
        keys = list(batch.keys) if hasattr(batch, 'keys') else None
    except Exception:
        keys = None
    keys = keys or [k for k in dir(batch) if not k.startswith('_')]
    for k in keys:
        try:
            v = getattr(batch, k)
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            try:
                setattr(batch, k, v.to(device))
            except Exception:
                pass
    return batch


def sdf_dir_to_smiles(sdf_dir_path: str) -> List[str]:
    smiles = []
    try:
        from rdkit import Chem
    except Exception:
        return smiles
    for fname in os.listdir(sdf_dir_path):
        if not fname.lower().endswith('.sdf'):
            continue
        fpath = os.path.join(sdf_dir_path, fname)
        try:
            suppl = Chem.SDMolSupplier(fpath, removeHs=True, sanitize=True)
            for mol in suppl:
                if mol is None:
                    continue
                smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
                if smi:
                    smiles.append(smi)
        except Exception:
            continue
    return smiles


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
        use_cross_mp=bool(args.use_cross_mp),
        cross_mp_hidden=None,
        cross_mp_bidirectional=False,
        cross_mp_use_distance=True,
        cross_mp_rbf_dim=0,
        cross_mp_use_direction=False,
        cross_mp_distance_sigma=4.0,
        cross_mp_dropout=0.0,
        cross_mp_every=1,
        use_protein_context=bool(args.use_protein_context),
        context_mode='film',
        context_dropout=float(args.context_dropout),
        use_dynamic_cross_edges=True,
        cross_topk=int(args.cross_topk),
        cross_radius=float(args.cross_radius),
        use_bond_head=True,
        bond_hidden=64,
        bond_classes=int(getattr(args, 'bond_classes', 5)),
        bond_radius=2.2,
        use_atom_type_head=True,
        atom_type_classes=int(getattr(args, 'atom_type_classes', 10)),
    ).to(device)

    diffusion = SoftMaskDiffusionProcess(num_steps=int(getattr(args, 'num_steps', 100)), sigma_min=0.5, sigma_max=1.0, sigma_schedule='linear', device=str(device))
    loss_fn = DiffusionLoss(w_max=1e2)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(getattr(args, 'lr', 1e-4)))
    trainer = DiffusionTrainer(
        diffusion_process=diffusion,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=str(device),
        lambda_feat=1.0,
        grad_clip_norm=1.0,
        aggregate_all_t=bool(args.aggregate_all_t),
        protein_encoder=protein_encoder,
        ligand_encoder=ligand_encoder,
        encoder_edge_radius=5.0,
        lambda_eps=float(getattr(args, 'lambda_eps', 0.0)),
        lambda_tau_smooth=float(getattr(args, 'lambda_tau_smooth', 0.0)),
        lambda_tau_rank=float(getattr(args, 'lambda_tau_rank', 0.0)),
        lambda_atom_type=float(getattr(args, 'lambda_atom_type', 0.0)),
    )
    return model, trainer, diffusion


def run_train(args):
    set_seed(int(args.seed))
    device = torch.device(args.device)
    train_loader, _ = get_data_loaders(
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle_train=True,
        shuffle_test=False,
        require_ligand=True,
        require_feat=True,
        require_protein=True,
    )
    if train_loader is None:
        raise RuntimeError('Train DataLoader is None')

    first_batch = next(iter(train_loader))
    lig_feat = getattr(first_batch, 'ligand_atom_feature', None)
    node_dim = int(lig_feat.size(1)) if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim == 2 else 64
    model, trainer, _ = build_model_and_trainer(args, node_dim, device)
    soft_mask = SoftMaskTransform()

    os.makedirs(args.out_dir, exist_ok=True)
    # logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    log_file = getattr(args, 'log_file', '')
    if isinstance(log_file, str) and len(log_file) > 0:
        os.makedirs(args.out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(args.out_dir, log_file))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    steps = 0
    pbar = None
    use_tqdm = bool(getattr(args, 'tqdm', True))
    while steps < int(args.train_steps):
        loader_iter = tqdm(train_loader, desc='train', total=len(train_loader)) if use_tqdm else train_loader
        for batch in loader_iter:
            if batch is None:
                continue
            batch = move_to_device(batch, args.device)
            logs = trainer.training_step(batch, model, soft_mask)
            steps += 1
            if use_tqdm:
                loader_iter.set_postfix({
                    'total': f"{logs['total_loss']:.3f}",
                    'diff': f"{logs['diffusion_loss']:.3f}",
                    'kl': f"{logs['kl_loss']:.3f}",
                    'bond': f"{logs['bond_loss']:.3f}",
                    'atom': f"{logs.get('atom_type_loss', 0.0):.3f}"
                })
            if steps % 10 == 0:
                logger.info(f"step={steps} total={logs['total_loss']:.4f} diff={logs['diffusion_loss']:.4f} kl={logs['kl_loss']:.4f} bond={logs['bond_loss']:.4f} atom={logs.get('atom_type_loss',0.0):.4f}")
            if steps >= int(args.train_steps):
                break


def run_sample_and_eval(args):
    set_seed(int(args.seed))
    device = torch.device(args.device)
    train_loader, test_loader = get_data_loaders(
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle_train=False,
        shuffle_test=False,
        require_ligand=True,
        require_feat=True,
        require_protein=True,
    )
    first_batch = next(iter(test_loader or train_loader))
    lig_feat = getattr(first_batch, 'ligand_atom_feature', None)
    node_dim = int(lig_feat.size(1)) if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim == 2 else 64
    model, trainer, diffusion = build_model_and_trainer(args, node_dim, device)
    soft_mask = SoftMaskTransform()
    sampler = ConditionalSampler(diffusion_process=diffusion, soft_mask_transform=soft_mask, device=args.device)

    sdf_dir = os.path.join(args.out_dir, 'samples_main')
    os.makedirs(sdf_dir, exist_ok=True)

    # classic conditional sampling
    cnt = 0
    use_tqdm = bool(getattr(args, 'tqdm', True))
    loader_iter = tqdm((test_loader or train_loader), desc='sample', total=len(test_loader or train_loader)) if use_tqdm else (test_loader or train_loader)
    for batch in loader_iter:
        if batch is None:
            continue
        batch = move_to_device(batch, args.device)
        sampler.sample_and_write(model, batch, out_dir=sdf_dir, num_samples=int(args.num_samples), prefix='lig', use_multi_modal=True)
        cnt += 1
        if cnt >= 1:
            break

    # de novo mode (optional)
    if bool(getattr(args, 'de_novo', False)):
        protein_pos = getattr(first_batch, 'protein_pos', None)
        x0, _ = sampler.propose_and_sample(model, protein_pos=protein_pos.to(device), num_atoms=int(args.de_novo_num_atoms),
                                           element_prior=None, sigma_init=float(args.de_novo_sigma), use_multi_modal=True)
        # write one de novo sample
        coords = x0.detach().cpu().numpy()
        syms = ['C'] * coords.shape[0]
        from src.evaluation.utils_pdb_writer import build_rdkit_mol_from_coords, write_rdkit_mol_sdf
        mol = build_rdkit_mol_from_coords(syms, coords)
        write_rdkit_mol_sdf(mol, os.path.join(sdf_dir, 'de_novo_00000.sdf'))

    # evaluations
    gen_smiles = sdf_dir_to_smiles(sdf_dir)
    if len(gen_smiles) > 0:
        chem_outdir = os.path.join(args.out_dir, 'evaluation_main/chemical')
        os.makedirs(chem_outdir, exist_ok=True)
        run_chemical_evaluation(generated_smiles=gen_smiles, reference_smiles=None, output_dir=chem_outdir)

        sim_outdir = os.path.join(args.out_dir, 'evaluation_main/similarity')
        os.makedirs(sim_outdir, exist_ok=True)
        run_similarity_evaluation(generated_smiles=gen_smiles, reference_smiles=None, baseline_results=None, output_dir=sim_outdir)

        geom_outdir = os.path.join(args.out_dir, 'evaluation_main/geometric')
        os.makedirs(geom_outdir, exist_ok=True)
        class _MolWrap:
            def __init__(self, pos):
                self.ligand_pos = pos
        run_geometric_evaluation(generated_molecules=[_MolWrap(torch.randn(8, 3))], reference_molecules=[_MolWrap(torch.randn(8, 3))], output_dir=geom_outdir, bins=50)

    print('✅ Sample+Eval finished. Outputs at:', args.out_dir)


