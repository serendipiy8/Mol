import os
from typing import List

import torch
import logging
from tqdm import tqdm

from src.model.utils.repro import set_seed
from src.data.data_loader import get_data_loaders
from src.data.soft_mask_transforms import SoftMaskTransform
from src.model.sampler import ConditionalSampler
from src.evaluation.geometric_evaluation import run_geometric_evaluation
from src.evaluation.chemical_evaluation import run_chemical_evaluation
from src.evaluation.similarity_evaluation import run_similarity_evaluation
from src.evaluation.molecular_properties import run_molecular_properties_evaluation
from src.evaluation.interactions_evaluation import run_interactions_evaluation
from src.evaluation.docking_evaluation import run_docking_evaluation
from src.evaluation.bond_evaluation import run_bond_evaluation
from .builders import build_model_and_trainer
from .utils import move_to_device, sdf_dir_to_smiles


 # Model Training
def run_train(args):
    import os
    import logging
    import torch
    from tqdm import tqdm

    set_seed(int(args.seed))
    device = torch.device(args.device)

    train_loader, _ = get_data_loaders(dataset_path=args.dataset_path, split_file=args.split_file,
                                       batch_size=int(args.batch_size), num_workers=int(args.num_workers),
                                       shuffle_train=False, shuffle_test=False, require_ligand=True,
                                       require_feat=True, require_protein=True,
                                       train_fraction=float(getattr(args, 'train_fraction', 1.0)),
                                       train_limit=int(getattr(args, 'train_limit', 0) or 0),
                                       test_limit=int(getattr(args, 'test_limit', 0) or 0))

    if train_loader is None:
        raise RuntimeError('Train DataLoader is None')

    first_batch = next(iter(train_loader))
    lig_feat = getattr(first_batch, 'ligand_atom_feature', None)
    node_dim = int(lig_feat.size(1)) if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim == 2 else 64

    # Force-enable protein context injection and cross-graph message passing for training
    model, trainer, _ = build_model_and_trainer(args, node_dim, device)
    soft_mask = SoftMaskTransform()

    os.makedirs(args.out_dir, exist_ok=True)

    # Logger
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
    use_tqdm = bool(getattr(args, 'tqdm', True))

    # Resume checkpoint
    if bool(getattr(args, 'resume', False)):
        ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
        ckpt_path = os.path.join(ckpt_dir, 'last.pt')
        if os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state.get('model', {}), strict=False)
                if 'optimizer' in state and hasattr(trainer, 'optimizer'):
                    trainer.optimizer.load_state_dict(state['optimizer'])
                steps = int(state.get('steps', 0))
                print(f"Resumed from {ckpt_path} at step {steps}")
            except Exception as e:
                print(f"Warning: failed to resume from {ckpt_path}: {e}")

    num_epochs = int(getattr(args, 'num_epochs', 1) or 1)
    train_steps = int(getattr(args, 'train_steps', 0) or 0)
    epoch_csv_path = os.path.join(args.out_dir, 'train_logs.csv')
    if not os.path.isfile(epoch_csv_path):
        with open(epoch_csv_path, 'w', encoding='utf-8') as f:
            f.write('epoch,total,diff,coord,feat,kl,bond,atom\n')

    # --------------------------
    # 每个 epoch 独立 tqdm
    for epoch in range(num_epochs):
        sum_total = sum_diff = sum_coord = sum_feat = sum_kl = sum_bond = sum_atom = 0.0
        sum_gcoord = sum_gfeat = 0.0
        batches_in_epoch = 0

        if use_tqdm:
            pbar = tqdm(total=len(train_loader),
                        desc=f"train e{epoch+1}/{num_epochs}",
                        dynamic_ncols=True,
                        leave=True)

        # helper: unwrap possible dict/list wrappers → Data/Batch
        def _unwrap_debug(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                for k in ('data', 'batch'):
                    if k in obj and obj[k] is not None:
                        return obj[k]
                for v in obj.values():
                    try:
                        if hasattr(v, 'keys') or hasattr(v, 'x') or hasattr(v, 'pos'):
                            return v
                    except Exception:
                        continue
                return obj
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    if it is not None:
                        return _unwrap_debug(it)
            return obj

        for bidx, batch in enumerate(train_loader):
            if batch is None:
                continue
            batch = move_to_device(batch, args.device)
            batch_dbg = _unwrap_debug(batch)

            bb = batch_dbg if batch_dbg is not None else batch
            logs = trainer.training_step(bb, model, soft_mask)
            steps += 1

            # 累计 loss
            sum_total += float(logs.get('total_loss', 0.0))
            sum_diff += float(logs.get('diffusion_loss', 0.0))
            sum_coord += float(logs.get('coord_loss', 0.0))
            sum_feat += float(logs.get('feat_loss', 0.0))
            sum_kl += float(logs.get('kl_loss', 0.0))
            sum_bond += float(logs.get('bond_loss', 0.0))
            sum_atom += float(logs.get('atom_type_loss', 0.0))
            # drop tau stats from epoch aggregation
            batches_in_epoch += 1
            # accumulate grad norms (post-backward, pre-step)
            sum_gcoord += float(logs.get('grad_coord', 0.0))
            sum_gfeat += float(logs.get('grad_feat', 0.0))

            if use_tqdm:
                pbar.update(1)

        if use_tqdm:
            pbar.close()

        # 每个 epoch 结束输出平均 loss
        if batches_in_epoch > 0:
            avg_total = sum_total / batches_in_epoch
            avg_diff = sum_diff / batches_in_epoch
            avg_coord = sum_coord / batches_in_epoch
            avg_feat = sum_feat / batches_in_epoch
            avg_kl = sum_kl / batches_in_epoch
            avg_bond = sum_bond / batches_in_epoch
            avg_atom = sum_atom / batches_in_epoch
            avg_gcoord = sum_gcoord / batches_in_epoch
            avg_gfeat = sum_gfeat / batches_in_epoch
            # drop tau stats from epoch aggregation

            logger.info(
                f"epoch={epoch+1}/{num_epochs} avg_total={avg_total:.4f} "
                f"avg_diff={avg_diff:.4f} avg_coord={avg_coord:.4f} avg_feat={avg_feat:.4f} "
                f"avg_kl={avg_kl:.4f} avg_bond={avg_bond:.4f} avg_atom={avg_atom:.4f} "
                f"avg_gcoord={avg_gcoord:.4e} avg_gfeat={avg_gfeat:.4e}"
            )

            # 写 CSV
            try:
                with open(epoch_csv_path, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch+1},{avg_total:.6f},{avg_diff:.6f},{avg_coord:.6f},{avg_feat:.6f},"
                            f"{avg_kl:.6f},{avg_bond:.6f},{avg_atom:.6f}\n")
            except Exception:
                pass

    try:
        ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, 'last.pt')

        torch.save({
            'model': model.state_dict(),
            'optimizer': getattr(trainer, 'optimizer', None).state_dict() if hasattr(trainer, 'optimizer') else None,
            'steps': steps,
            'args': vars(args)
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

    except Exception as e:
        print(f"Warning: failed to save checkpoint: {e}")


    try:
        import csv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = []
        total_vals = []
        diff_vals = []
        kl_vals = []
        bond_vals = []
        atom_vals = []

        if os.path.isfile(epoch_csv_path):
            with open(epoch_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        epochs.append(int(row.get('epoch', 0)))
                        total_vals.append(float(row.get('total', 0.0)))
                        diff_vals.append(float(row.get('diff', 0.0)))
                        kl_vals.append(float(row.get('kl', 0.0)))
                        bond_vals.append(float(row.get('bond', 0.0)))
                        atom_vals.append(float(row.get('atom', 0.0)))
                    except Exception:
                        continue

        if len(epochs) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, total_vals, label='total', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (avg per epoch)')
            plt.title('Training Loss per Epoch')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_path = os.path.join(args.out_dir, 'train_loss_plot.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            print(f"Saved loss plot to {plot_path}")
            plt.close()
    except Exception as e:
        print(f"Warning: failed to plot train losses: {e}")


# Model Sampling and Evaluation
def run_sample_conditional(args):

    set_seed(int(args.seed))
    device = torch.device(args.device)
    train_loader, test_loader = get_data_loaders(dataset_path=args.dataset_path, split_file=args.split_file,
                                                 batch_size=int(args.batch_size), num_workers=int(args.num_workers),
                                                 shuffle_train=False, shuffle_test=False, require_ligand=True,
                                                 require_feat=True, require_protein=True)

    first_batch = next(iter(test_loader or train_loader))
    lig_feat = getattr(first_batch, 'ligand_atom_feature', None)
    node_dim = int(lig_feat.size(1)) if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim == 2 else 64
    model, trainer, diffusion = build_model_and_trainer(args, node_dim, device)

    ckpt_path = str(getattr(args, 'ckpt', '') or '').strip()

    if len(ckpt_path) == 0:
        maybe_path = os.path.join(args.out_dir, 'checkpoints', 'last.pt')
        if os.path.isfile(maybe_path):
            ckpt_path = maybe_path

    if len(ckpt_path) > 0 and os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state.get('model', {}), strict=False)
            print(f"Loaded checkpoint for sampling: {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")
            
    soft_mask = SoftMaskTransform()
    sampler = ConditionalSampler(diffusion_process=diffusion, soft_mask_transform=soft_mask, device=args.device)

    sdf_dir = os.path.join(args.out_dir, 'samples_conditional')
    os.makedirs(sdf_dir, exist_ok=True)

    # classic conditional sampling
    cnt = 0
    use_tqdm = bool(getattr(args, 'tqdm', True))
    loader_iter = tqdm((test_loader or train_loader), desc='sample', total=len(test_loader or train_loader)) if use_tqdm else (test_loader or train_loader)

    # helper: unwrap possible dict/list wrappers → Data/Batch
    def _unwrap_debug(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            for k in ('data', 'batch'):
                if k in obj and obj[k] is not None:
                    return obj[k]
            for v in obj.values():
                try:
                    if hasattr(v, 'x') or hasattr(v, 'pos') or hasattr(v, 'ligand_pos'):
                        return v
                except Exception:
                    continue
            return obj
        if isinstance(obj, (list, tuple)):
            for it in obj:
                if it is not None:
                    return _unwrap_debug(it)
        return obj
    for batch in loader_iter:
        if batch is None:
            continue
        batch = move_to_device(batch, args.device)
        bb = _unwrap_debug(batch)
        # edge-free sampling flag
        if bool(getattr(args, 'edge_free_sampling', False)):
            try:
                setattr(bb, '_edge_free', True)
            except Exception:
                pass
        # Write reference from the actual batch we are sampling (only once)
        if cnt == 0:
            try:
                ref_dir = os.path.join('experiments', 'reference')
                os.makedirs(ref_dir, exist_ok=True)
                lig_pos = getattr(bb, 'ligand_pos', None)
                lig_el = getattr(bb, 'ligand_element', None)
                if isinstance(lig_pos, torch.Tensor) and lig_pos.ndim == 2 and lig_pos.size(0) > 0 and lig_pos.size(1) == 3:
                    coords = lig_pos.detach().cpu().numpy()
                    symbols = None
                    if isinstance(lig_el, torch.Tensor) and lig_el.numel() == lig_pos.size(0):
                        try:
                            from rdkit import Chem
                            pt = Chem.GetPeriodicTable()
                            z = lig_el.detach().cpu().to(torch.long).numpy().tolist()
                            symbols = [pt.GetElementSymbol(int(max(1, zi))) if int(zi) > 0 else 'C' for zi in z]
                        except Exception:
                            pass
                    if symbols is None:
                        symbols = ['C'] * coords.shape[0]
                    from src.evaluation.utils_pdb_writer import build_rdkit_mol_from_coords, write_rdkit_mol_sdf
                    mol = build_rdkit_mol_from_coords(symbols, coords)
                    ref_path = os.path.join(ref_dir, 'reference_00000.sdf')
                    write_rdkit_mol_sdf(mol, ref_path)
                    print(f"Saved reference ligand SDF to {ref_path}")
            except Exception as e:
                print(f"Warning: failed to write reference ligand SDF: {e}")
        sampler.sample_and_write(model, bb, out_dir=sdf_dir, num_samples=int(args.num_samples), prefix='lig', use_multi_modal=True)
        cnt += 1
        if cnt >= 1:
            break

    # Compare generated vs reference coordinates (if available)
    try:
        ref_dir = os.path.join('experiments', 'reference')
        ref_path = os.path.join(ref_dir, 'reference_00000.sdf')
        gen0_path = os.path.join(sdf_dir, 'lig_00000.sdf')
        if os.path.isfile(ref_path) and os.path.isfile(gen0_path):
            from rdkit import Chem
            import numpy as np
            def _load_coords(sdf_path):
                suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
                for m in suppl:
                    if m is None:
                        continue
                    if m.GetNumConformers() == 0:
                        continue
                    conf = m.GetConformer()
                    n = m.GetNumAtoms()
                    xyz = np.zeros((n, 3), dtype=float)
                    for i in range(n):
                        p = conf.GetAtomPosition(i)
                        xyz[i, 0] = p.x; xyz[i, 1] = p.y; xyz[i, 2] = p.z
                    return xyz
                return None
            X = _load_coords(ref_path)
            Y = _load_coords(gen0_path)
            if X is not None and Y is not None and X.shape[0] == Y.shape[0] and X.shape[0] > 0:
                # raw stats
                def _stats(Z):
                    return dict(mean=Z.mean(axis=0).tolist(), std=Z.std(axis=0).tolist(), min=Z.min(axis=0).tolist(), max=Z.max(axis=0).tolist())
                # center
                Xc = X - X.mean(axis=0, keepdims=True)
                Yc = Y - Y.mean(axis=0, keepdims=True)
                # Kabsch
                H = Xc.T @ Yc
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt2 = Vt.copy(); Vt2[-1, :] *= -1.0
                    R = Vt2.T @ U.T
                Yc_aligned = (Yc @ R.T)
                rmsd_raw = float(np.sqrt(((X - Y) ** 2).sum(axis=1).mean()))
                rmsd_centered = float(np.sqrt(((Xc - Yc) ** 2).sum(axis=1).mean()))
                rmsd_kabsch = float(np.sqrt(((Xc - Yc_aligned) ** 2).sum(axis=1).mean()))
                print('[REF-COMP] N_atoms=', X.shape[0], 'raw_RMSD=', rmsd_raw, 'centered_RMSD=', rmsd_centered, 'kabsch_RMSD=', rmsd_kabsch)
            else:
                print('[REF-COMP] Skip: coord load failed or atom counts mismatch.')
        else:
            print('[REF-COMP] Skip: missing ref/gen files.')
    except Exception as e:
        print(f"[REF-COMP] comparison failed: {e}")

    # evaluations (full chain when possible)
    gen_smiles = sdf_dir_to_smiles(sdf_dir)
    if len(gen_smiles) > 0:
        # chemical
        chem_outdir = os.path.join(args.out_dir, 'evaluation_conditional/chemical')
        os.makedirs(chem_outdir, exist_ok=True)
        run_chemical_evaluation(generated_smiles=gen_smiles, reference_smiles=None, output_dir=chem_outdir)

        # similarity
        sim_outdir = os.path.join(args.out_dir, 'evaluation_conditional/similarity')
        os.makedirs(sim_outdir, exist_ok=True)
        run_similarity_evaluation(generated_smiles=gen_smiles, reference_smiles=None, baseline_results=None, output_dir=sim_outdir)

        # molecular properties（参考集缺失时仅导出生成分子的统计对比为空）
        prop_outdir = os.path.join(args.out_dir, 'evaluation_conditional/molecular_properties')
        os.makedirs(prop_outdir, exist_ok=True)
        run_molecular_properties_evaluation(generated_smiles=gen_smiles, reference_smiles=[], output_dir=prop_outdir, properties=None)

        # bond-level evaluation
        bond_outdir = os.path.join(args.out_dir, 'evaluation_conditional/bond')
        os.makedirs(bond_outdir, exist_ok=True)
        run_bond_evaluation(generated_sdf_dir=sdf_dir, output_dir=bond_outdir)

        # geometric placeholder（需要参考结构，这里示例用随机）
        geom_outdir = os.path.join(args.out_dir, 'evaluation_conditional/geometric')
        os.makedirs(geom_outdir, exist_ok=True)
        class _MolWrap:
            def __init__(self, pos):
                self.ligand_pos = pos
        run_geometric_evaluation(generated_molecules=[_MolWrap(torch.randn(8, 3))], reference_molecules=[_MolWrap(torch.randn(8, 3))], output_dir=geom_outdir, bins=50)

    print('✅ Conditional sampling finished. Outputs at:', sdf_dir)


def run_sample_de_novo(args):

    set_seed(int(args.seed))
    device = torch.device(args.device)
    train_loader, test_loader = get_data_loaders(dataset_path=args.dataset_path, split_file=args.split_file,
                                                 batch_size=int(args.batch_size), num_workers=int(args.num_workers),
                                                 shuffle_train=False, shuffle_test=False, require_ligand=True,
                                                 require_feat=True, require_protein=True)

    first_batch = next(iter(test_loader or train_loader))
    lig_feat = getattr(first_batch, 'ligand_atom_feature', None)
    node_dim = int(lig_feat.size(1)) if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim == 2 else 64
    model, trainer, diffusion = build_model_and_trainer(args, node_dim, device)

    ckpt_path = str(getattr(args, 'ckpt', '') or '').strip()
    if len(ckpt_path) == 0:
        maybe_path = os.path.join(args.out_dir, 'checkpoints', 'last.pt')
        if os.path.isfile(maybe_path):
            ckpt_path = maybe_path
    if len(ckpt_path) > 0 and os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state.get('model', {}), strict=False)
            print(f"Loaded checkpoint for sampling: {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")

    soft_mask = SoftMaskTransform()
    sampler = ConditionalSampler(diffusion_process=diffusion, soft_mask_transform=soft_mask, device=args.device)

    sdf_dir = os.path.join(args.out_dir, 'samples_de_novo')
    os.makedirs(sdf_dir, exist_ok=True)

    protein_pos = getattr(first_batch, 'protein_pos', None)
    if protein_pos is None:
        raise RuntimeError('protein_pos missing in batch for de novo sampling')

    # produce multiple molecules de novo
    for i in range(int(args.num_samples)):
        x0, _ = sampler.propose_and_sample(
            model,
            protein_pos=protein_pos.to(device),
            num_atoms=int(args.de_novo_num_atoms),
            element_prior=None,
            sigma_init=float(args.de_novo_sigma),
            use_multi_modal=True
        )
        coords = x0.detach().cpu().numpy()
        syms = ['C'] * coords.shape[0]
        from src.evaluation.utils_pdb_writer import build_rdkit_mol_from_coords, write_rdkit_mol_sdf
        mol = build_rdkit_mol_from_coords(syms, coords)
        write_rdkit_mol_sdf(mol, os.path.join(sdf_dir, f'de_novo_{i:05d}.sdf'))

    # evaluations (chem + sim + properties + bond; geom/docking/interactions 视可用数据)
    gen_smiles = sdf_dir_to_smiles(sdf_dir)
    if len(gen_smiles) > 0:
        # chemical
        chem_outdir = os.path.join(args.out_dir, 'evaluation_de_novo/chemical')
        os.makedirs(chem_outdir, exist_ok=True)
        run_chemical_evaluation(generated_smiles=gen_smiles, reference_smiles=None, output_dir=chem_outdir)

        # similarity
        sim_outdir = os.path.join(args.out_dir, 'evaluation_de_novo/similarity')
        os.makedirs(sim_outdir, exist_ok=True)
        run_similarity_evaluation(generated_smiles=gen_smiles, reference_smiles=None, baseline_results=None, output_dir=sim_outdir)

        # molecular properties
        prop_outdir = os.path.join(args.out_dir, 'evaluation_de_novo/molecular_properties')
        os.makedirs(prop_outdir, exist_ok=True)
        run_molecular_properties_evaluation(generated_smiles=gen_smiles, reference_smiles=[], output_dir=prop_outdir, properties=None)

        # bond-level evaluation
        # bond_outdir = os.path.join(args.out_dir, 'evaluation_de_novo/bond')
        # os.makedirs(bond_outdir, exist_ok=True)
        # run_bond_evaluation(generated_sdf_dir=sdf_dir, output_dir=bond_outdir)

    print('✅ De novo sampling finished. Outputs at:', sdf_dir)


def run_sample_and_eval(args):
    run_sample_conditional(args)
    if bool(getattr(args, 'de_novo', False)):
        run_sample_de_novo(args)

    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        vis_dir = os.path.join(args.out_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)


        sdf_paths = []
        cond_dir = os.path.join(args.out_dir, 'samples_conditional')
        de_dir = os.path.join(args.out_dir, 'samples_de_novo')
        for d in (cond_dir, de_dir):
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.lower().endswith('.sdf'):
                        sdf_paths.append(os.path.join(d, fn))

        mols = []
        for p in sdf_paths[:64]:  # 限制网格显示前 64 个
            try:
                suppl = Chem.SDMolSupplier(p, removeHs=True, sanitize=True)
                for mol in suppl:
                    if mol is not None:
                        mols.append(mol)
                        break
            except Exception:
                continue
        if len(mols) > 0:
            img = Draw.MolsToGridImage(mols, molsPerRow=8, subImgSize=(200, 200))
            vis_path = os.path.join(vis_dir, 'vis_grid.png')
            img.save(vis_path)
            print(f"Saved molecule grid image to {vis_path}")

        # 写出简要摘要
        import json
        summary = {
            'num_conditional': len(os.listdir(cond_dir)) if os.path.isdir(cond_dir) else 0,
            'num_de_novo': len(os.listdir(de_dir)) if os.path.isdir(de_dir) else 0,
            'num_visualized': len(mols)
        }
        with open(os.path.join(args.out_dir, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Warning: post-visualization failed: {e}")


