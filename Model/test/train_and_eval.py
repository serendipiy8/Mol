import os
import sys
import argparse
from typing import Optional, List

import torch


def add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


add_src_to_path()

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


def move_batch_to_device(batch, device: str):
    # dict-like: recursively move inner payloads too
    if isinstance(batch, dict):
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, (dict,)):
                batch[k] = move_batch_to_device(v, device)
        # try common nested keys
        for k in ('data', 'batch'):
            if k in batch and batch[k] is not None:
                inner = batch[k]
                if isinstance(inner, (dict,)):
                    batch[k] = move_batch_to_device(inner, device)
                else:
                    try:
                        keys = list(inner.keys) if hasattr(inner, 'keys') else None
                    except Exception:
                        keys = None
                    if keys is not None:
                        for name in keys:
                            val = getattr(inner, name)
                            if isinstance(val, torch.Tensor):
                                setattr(inner, name, val.to(device))
        return batch
    # PyG Batch/Data: prefer .keys if available
    try:
        keys = list(batch.keys) if hasattr(batch, 'keys') else None
    except Exception:
        keys = None
    if keys is None:
        # Fallback: iterate attributes but skip private
        keys = [k for k in dir(batch) if not k.startswith('_')]
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


def main():
    parser = argparse.ArgumentParser(description='Train and Evaluate (Main Experiment)')
    parser.add_argument('--dataset_path', type=str, default='src/data/crossdocked_v1.1_rmsd1.0_processed', help='LMDB dataset folder')
    parser.add_argument('--split_file', type=str, default='src/data/split_by_name.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='Model/experiments/outputs/train_and_eval')
    parser.add_argument('--num_samples', type=int, default=4)
    # model/config switches
    parser.add_argument('--use_cross_mp', action='store_true', default=True)
    parser.add_argument('--cross_radius', type=float, default=6.0)
    parser.add_argument('--cross_topk', type=int, default=10)
    parser.add_argument('--use_protein_context', action='store_true', default=True)
    parser.add_argument('--context_dropout', type=float, default=0.0)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--aggregate_all_t', action='store_true', default=False)
    # logging/checkpoint
    parser.add_argument('--log_csv', action='store_true', default=True)
    parser.add_argument('--ckpt_every', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')
    # docking eval (optional)
    parser.add_argument('--run_docking', action='store_true', default=False)
    parser.add_argument('--vina_path', type=str, default='vina')
    parser.add_argument('--receptor_pdbqt', type=str, nargs='*', default=[])
    parser.add_argument('--receptor_pdb', type=str, default='')
    parser.add_argument('--center', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument('--size', type=float, nargs=3, default=[20.0, 20.0, 20.0])
    # trainer regularizers
    parser.add_argument('--lambda_eps', type=float, default=0.0)
    parser.add_argument('--lambda_tau_smooth', type=float, default=0.0)
    parser.add_argument('--lambda_tau_rank', type=float, default=0.0)
    # encoder options
    parser.add_argument('--protein_encoder_se3', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    # Data
    train_loader, test_loader = get_data_loaders(
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=True,
        shuffle_test=False,
        require_ligand=True,
        require_feat=True,
        require_protein=True,
    )
    if train_loader is None:
        raise RuntimeError('Train DataLoader is None')

    # Peek a batch to infer dims
    first_batch = None
    for b in train_loader:
        if b is not None:
            first_batch = b
            break
    if first_batch is None:
        raise RuntimeError('No valid batch found')

    lig_feat = getattr(first_batch, 'ligand_atom_feature', None)
    node_dim = int(lig_feat.size(1)) if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim == 2 else 64

    device = torch.device(args.device)

    # Model components
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
        bond_classes=2,
        bond_radius=2.2,
    ).to(device)

    diffusion = SoftMaskDiffusionProcess(num_steps=100, sigma_min=0.5, sigma_max=1.0, sigma_schedule='linear', device=args.device)
    loss_fn = DiffusionLoss(w_max=1e2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DiffusionTrainer(
        diffusion_process=diffusion,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
        lambda_feat=1.0,
        grad_clip_norm=1.0,
        aggregate_all_t=bool(args.aggregate_all_t),
        protein_encoder=protein_encoder,
        ligand_encoder=ligand_encoder,
        encoder_edge_radius=5.0,
        lambda_eps=float(args.lambda_eps),
        lambda_tau_smooth=float(args.lambda_tau_smooth),
        lambda_tau_rank=float(args.lambda_tau_rank),
    )

    soft_mask = SoftMaskTransform()

    # Optional resume
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state.get('model', {}), strict=False)
        optimizer.load_state_dict(state.get('optim', {}))

    # Train (short)
    model.train()
    steps = 0
    csv_f = None
    if args.log_csv:
        import csv
        csv_path = os.path.join(args.out_dir, 'train_log.csv')
        csv_f = open(csv_path, 'w', newline='')
        csv_w = csv.writer(csv_f)
        csv_w.writerow(['step','total_loss','diffusion_loss','kl_loss','bond_loss'])
    while steps < args.train_steps:
        for batch in train_loader:
            if batch is None:
                continue
            batch = move_batch_to_device(batch, args.device)
            logs = trainer.training_step(batch, model, soft_mask)
            steps += 1
            if steps % 10 == 0:
                print(f"step={steps} | total={logs['total_loss']:.4f} diff={logs['diffusion_loss']:.4f} kl={logs['kl_loss']:.4f} bond={logs['bond_loss']:.4f}")
            if csv_f is not None:
                csv_w.writerow([steps, logs['total_loss'], logs['diffusion_loss'], logs['kl_loss'], logs['bond_loss']])
                csv_f.flush()
            if args.ckpt_every and steps % int(args.ckpt_every) == 0:
                ckpt = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': steps,
                    'args': vars(args),
                }
                torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_step_{steps}.pt'))
            if steps >= args.train_steps:
                break

    # Sample a few molecules from test (or train) and write SDF
    model.eval()
    sampler = ConditionalSampler(diffusion_process=diffusion, soft_mask_transform=soft_mask, device=args.device)
    eval_batch_list: List[torch.nn.Module] = []
    if test_loader is not None:
        for batch in test_loader:
            if batch is not None:
                eval_batch_list.append(batch)
                if len(eval_batch_list) >= 1:
                    break
    if not eval_batch_list:
        eval_batch_list.append(first_batch)

    sdf_dir = os.path.join(args.out_dir, 'samples')
    os.makedirs(sdf_dir, exist_ok=True)

    gen_positions: List[torch.Tensor] = []
    ref_positions: List[torch.Tensor] = []

    for b in eval_batch_list:
        b = move_batch_to_device(b, args.device)
        # reference ligand positions from dataset
        if hasattr(b, 'ligand_pos') and isinstance(b.ligand_pos, torch.Tensor):
            ref_positions.append(b.ligand_pos.detach().cpu())
        # generate samples and write SDF
        sdf_paths = sampler.sample_and_write(model, b, out_dir=sdf_dir, num_samples=args.num_samples, prefix='lig', use_multi_modal=False)
        # we also keep their coordinates in memory for geometric evaluation
        # sample_once returns [N,3] on device; reuse it here for N samples
        for _ in range(args.num_samples):
            x0 = sampler.sample_once(model, b, use_multi_modal=False)
            gen_positions.append(x0.detach().cpu())

    # Geometric evaluation (positions only)
    # Wrap positions into minimal objects with .ligand_pos
    class _MolWrapper:
        def __init__(self, pos: torch.Tensor):
            self.ligand_pos = pos

    generated_mols = [_MolWrapper(p) for p in gen_positions]
    reference_mols = [_MolWrapper(p) for p in ref_positions]

    geom_outdir = os.path.join(args.out_dir, 'evaluation/geometric')
    os.makedirs(geom_outdir, exist_ok=True)
    run_geometric_evaluation(generated_molecules=generated_mols, reference_molecules=reference_mols, output_dir=geom_outdir, bins=50)

    # Chemical & Similarity evaluation from generated SDFs → SMILES (best-effort)
    def _sdf_dir_to_smiles(sdf_dir_path: str):
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

    gen_smiles = _sdf_dir_to_smiles(sdf_dir)
    if len(gen_smiles) > 0:
        chem_outdir = os.path.join(args.out_dir, 'evaluation/chemical')
        os.makedirs(chem_outdir, exist_ok=True)
        run_chemical_evaluation(generated_smiles=gen_smiles, reference_smiles=None, output_dir=chem_outdir)

        sim_outdir = os.path.join(args.out_dir, 'evaluation/similarity')
        os.makedirs(sim_outdir, exist_ok=True)
        run_similarity_evaluation(generated_smiles=gen_smiles, reference_smiles=None, baseline_results=None, output_dir=sim_outdir)

        # Docking evaluation (optional)
        if args.run_docking and len(args.receptor_pdbqt) > 0:
            from src.evaluation.docking_evaluation import run_docking_evaluation, DockingEvaluator
            docking_outdir = os.path.join(args.out_dir, 'evaluation/docking')
            os.makedirs(docking_outdir, exist_ok=True)
            # build binding_sites list for each receptor
            cx, cy, cz = args.center
            sx, sy, sz = args.size
            binding_sites = [{'center_x': cx, 'center_y': cy, 'center_z': cz, 'size_x': sx, 'size_y': sy, 'size_z': sz}
                             for _ in args.receptor_pdbqt]
            # set vina path
            # run_docking_evaluation constructs its own evaluator; we can set env Vina path by monkey-patching if needed
            # Here we call directly and rely on PATH or provide vina_path separately by updating class after import
            # Execute
            run_docking_evaluation(
                smiles_list=gen_smiles,
                receptor_paths=args.receptor_pdbqt,
                binding_sites=binding_sites,
                output_dir=docking_outdir,
            )

    if csv_f is not None:
        csv_f.close()
    # Interactions evaluation (optional receptor PDB)
    if getattr(args, 'receptor_pdb', ''):
        from src.evaluation.interactions_evaluation import run_interactions_evaluation
        inter_outdir = os.path.join(args.out_dir, 'evaluation/interactions')
        os.makedirs(inter_outdir, exist_ok=True)
        run_interactions_evaluation(generated_sdf_dir=sdf_dir, receptor_pdb_path=args.receptor_pdb, output_dir=inter_outdir)

    print('✅ Train + Sample + Geometric/Chemical/Similarity/Interactions Eval finished. Outputs at:', args.out_dir)


if __name__ == '__main__':
    main()


