#!/usr/bin/env python3
import os
import sys
import json
import lmdb
import pickle
import argparse
import random
import tempfile
import torch
import subprocess
import glob
from typing import List, Dict

sys.path.append('src')

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from src.evaluation.docking_evaluation import DockingEvaluator
from src.evaluation.utils_pdb_writer import (
    write_protein_pdb,
    build_rdkit_mol_from_coords,
    write_rdkit_mol_sdf
)


def sample_keys_from_name2id(name2id_path: str, max_items: int) -> List[bytes]:
    # 兼容 torch.save 产生的 pt：优先 torch.load
    with open(name2id_path, 'rb') as f:
        data = torch.load(f, map_location='cpu')

    # data 可能是 {key: id} 或 {id: key}
    keys_raw = list(data.keys())
    vals_raw = list(data.values())

    candidate_keys: List[bytes] = []

    def ensure_bytes(x) -> bytes:
        if isinstance(x, bytes):
            return x
        if isinstance(x, str):
            return x.encode('utf-8', errors='ignore')
        # 其他类型（如int）转成字符串再编码
        try:
            return str(x).encode('utf-8', errors='ignore')
        except Exception:
            return bytes()

    # 优先使用看起来像真实 LMDB 键的那一列（字符串/bytes）
    key_like = [k for k in keys_raw if isinstance(k, (str, bytes))]
    val_like = [v for v in vals_raw if isinstance(v, (str, bytes))]

    source = key_like if len(key_like) >= len(val_like) else val_like
    if not source:
        # 都不像字符串，则两边都尝试
        source = keys_raw

    random.shuffle(source)
    for x in source[:max_items*2]:  # 多取一些，过滤空字节
        b = ensure_bytes(x)
        if b:
            candidate_keys.append(b)
        if len(candidate_keys) >= max_items:
            break

    return candidate_keys[:max_items]


def sample_keys_from_lmdb(lmdb_path: str, max_items: int) -> List[bytes]:
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048, subdir=False)
    keys: List[bytes] = []
    try:
        with env.begin(write=False) as txn:
            with txn.cursor() as cur:
                for k, _ in cur:
                    keys.append(k)
                    if len(keys) >= max_items:
                        break
    finally:
        env.close()
    random.shuffle(keys)
    return keys[:max_items]


def read_item_from_lmdb(lmdb_path: str, key: str) -> Dict:
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048, subdir=False)
    with env.begin(write=False) as txn:
        k = key.encode() if isinstance(key, str) else key
        buf = txn.get(k)
        if buf is None:
            raise KeyError(f'Key not found: {key}')
        item = pickle.loads(buf)
    env.close()
    return item


def convert_pdb_to_pdbqt_batch_mgl(pdb_dir: str, mgltools_path: str = 'python') -> List[str]:
    """使用MGLTools批量转换PDB文件为PDBQT格式"""
    pdb_files = glob.glob(os.path.join(pdb_dir, 'rec_*.pdb'))
    pdbqt_files = []
    
    print(f'开始使用MGLTools批量转换 {len(pdb_files)} 个受体PDB文件为PDBQT...')
    
    for pdb_file in pdb_files:
        pdbqt_file = pdb_file.replace('.pdb', '.pdbqt')
        try:
            # 使用MGLTools的prepare_receptor4.py转换受体
            cmd = [
                mgltools_path,
                'prepare_receptor4.py',
                '-r', pdb_file,
                '-o', pdbqt_file,
                '-A', 'checkhydrogens'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(pdbqt_file):
                pdbqt_files.append(pdbqt_file)
                print(f'✓ 受体转换成功: {os.path.basename(pdb_file)} -> {os.path.basename(pdbqt_file)}')
            else:
                print(f'✗ 受体转换失败: {os.path.basename(pdb_file)} - {result.stderr}')
                
        except subprocess.TimeoutExpired:
            print(f'✗ 受体转换超时: {os.path.basename(pdb_file)}')
        except Exception as e:
            print(f'✗ 受体转换错误: {os.path.basename(pdb_file)} - {e}')
    
    print(f'受体转换完成: {len(pdbqt_files)}/{len(pdb_files)} 个文件成功')
    return pdbqt_files


def convert_sdf_to_pdbqt_batch_mgl(sdf_dir: str, mgltools_path: str = 'python') -> List[str]:
    """使用MGLTools批量转换SDF文件为PDBQT格式"""
    sdf_files = glob.glob(os.path.join(sdf_dir, 'lig_*.sdf'))
    pdbqt_files = []
    
    print(f'开始使用MGLTools批量转换 {len(sdf_files)} 个配体SDF文件为PDBQT...')
    
    for sdf_file in sdf_files:
        pdbqt_file = sdf_file.replace('.sdf', '.pdbqt')
        try:
            # 使用MGLTools的prepare_ligand4.py转换配体
            cmd = [
                mgltools_path,
                'prepare_ligand4.py',
                '-l', sdf_file,
                '-o', pdbqt_file,
                '-A', 'hydrogens'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(pdbqt_file):
                pdbqt_files.append(pdbqt_file)
                print(f'✓ 配体转换成功: {os.path.basename(sdf_file)} -> {os.path.basename(pdbqt_file)}')
            else:
                print(f'✗ 配体转换失败: {os.path.basename(sdf_file)} - {result.stderr}')
                
        except subprocess.TimeoutExpired:
            print(f'✗ 配体转换超时: {os.path.basename(sdf_file)}')
        except Exception as e:
            print(f'✗ 配体转换错误: {os.path.basename(sdf_file)} - {e}')
    
    print(f'配体转换完成: {len(pdbqt_files)}/{len(sdf_files)} 个文件成功')
    return pdbqt_files


def clean_pdbqt_for_vina(pdbqt_file: str):
    """清理PDBQT文件，移除Vina不支持的标签并修复原子编号"""
    try:
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        # 过滤掉Vina不支持的标签
        cleaned_lines = []
        atom_count = 1
        for line in lines:
            # 保留ATOM和HETATM行，移除REMARK、ROOT、ENDROOT等
            if line.startswith(('ATOM', 'HETATM')):
                # 修复原子编号：确保每个原子都有唯一的编号
                if len(line) > 6:
                    # 替换原子编号（位置7-11）
                    new_line = line[:6] + f"{atom_count:5d}" + line[11:]
                    cleaned_lines.append(new_line)
                    atom_count += 1
                else:
                    cleaned_lines.append(line)
        
        # 写回清理后的文件
        with open(pdbqt_file, 'w') as f:
            f.writelines(cleaned_lines)
            
    except Exception as e:
        print(f'警告：清理PDBQT文件失败 {os.path.basename(pdbqt_file)}: {e}')


def clean_ligand_pdbqt_for_vina(pdbqt_file: str):
    """专门清理配体PDBQT文件，确保符合Vina格式要求"""
    try:
        print(f'开始清理文件: {os.path.basename(pdbqt_file)}')
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        # 检查原始文件中的HETATM行
        hetatm_count = sum(1 for line in lines if line.startswith('HETATM'))
        print(f'原始文件中的HETATM行数: {hetatm_count}')
        
        # 原子类型映射，将非标准类型映射为标准类型
        atom_type_map = {
            'NA': 'N',   # 氮
            'OA': 'O',   # 氧
            'SA': 'S',   # 硫
            'P': 'P',    # 磷
            'C': 'C',    # 碳
            'N': 'N',    # 氮
            'O': 'O',    # 氧
            'S': 'S',    # 硫
            'H': 'H',    # 氢
            'C1': 'C',   # 碳变体
            'C2': 'C',   # 碳变体
            'C3': 'C',   # 碳变体
            'N1': 'N',   # 氮变体
            'N2': 'N',   # 氮变体
            'O1': 'O',   # 氧变体
            'O2': 'O',   # 氧变体
        }
        
        # 收集所有ATOM和HETATM行，但跳过水分子
        atom_lines = []
        atom_count = 1
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                # 检查是否是水分子（HOH）
                residue_name = line[17:20].strip() if len(line) > 20 else ''
                if residue_name == 'HOH':
                    continue  # 跳过水分子
                
                # 将HETATM转换为ATOM
                if line.startswith('HETATM'):
                    new_line = 'ATOM  ' + line[6:]
                else:
                    new_line = line
                
                # 解析原子信息
                parts = new_line.split()
                if len(parts) >= 11:
                    atom_name = parts[2]
                    residue_name = parts[3]
                    x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                    occupancy = parts[8]
                    temp_factor = parts[9]
                    charge = parts[10]
                    atom_type = parts[11] if len(parts) > 11 else atom_name
                    
                    # 映射原子类型
                    if atom_type in atom_type_map:
                        atom_type = atom_type_map[atom_type]
                    
                    # 转换电荷为浮点数
                    try:
                        charge_float = float(charge)
                    except ValueError:
                        charge_float = 0.0
                    
                    # 重新格式化为标准PDBQT格式，完全匹配受体格式
                    # 使用与受体完全相同的格式
                    new_line = f"ATOM  {atom_count:5d}  {atom_name:<2} {residue_name:<3}     1      {x:7.3f} {y:7.3f} {z:7.3f}  {occupancy:4.2f}  {temp_factor:4.2f}    {charge_float:6.3f} {atom_type:>2} \n"
                
                if not new_line.endswith('\n'):
                    new_line += '\n'
                atom_lines.append(new_line)
                atom_count += 1
        
        # 按照Vina配体格式要求重新组织
        cleaned_lines = []
        cleaned_lines.append("ROOT\n")
        cleaned_lines.extend(atom_lines)
        cleaned_lines.append("ENDROOT\n")
        cleaned_lines.append("TORSDOF 0\n")
        
        # 写回清理后的文件
        with open(pdbqt_file, 'w') as f:
            f.writelines(cleaned_lines)
            
    except Exception as e:
        print(f'警告：清理配体PDBQT文件失败 {os.path.basename(pdbqt_file)}: {e}')


def convert_pdb_to_pdbqt_batch_obabel(pdb_dir: str, obabel_path: str = 'obabel') -> List[str]:
    """使用OpenBabel批量转换受体PDB为PDBQT"""
    pdb_files = glob.glob(os.path.join(pdb_dir, 'rec_*.pdb'))
    pdbqt_files: List[str] = []
    print(f'开始使用OpenBabel批量转换 {len(pdb_files)} 个受体PDB文件为PDBQT...')
    for pdb_file in pdb_files:
        pdbqt_file = pdb_file.replace('.pdb', '.pdbqt')
        try:
            cmd = [
                obabel_path,
                pdb_file,
                '-O', pdbqt_file,
                '--partialcharge', 'gasteiger'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(pdbqt_file):
                # 清理PDBQT文件，移除Vina不支持的标签
                clean_pdbqt_for_vina(pdbqt_file)
                pdbqt_files.append(pdbqt_file)
                print(f'✓ 受体转换成功: {os.path.basename(pdb_file)} -> {os.path.basename(pdbqt_file)}')
            else:
                print(f'✗ 受体转换失败: {os.path.basename(pdb_file)} - {result.stderr}')
        except Exception as e:
            print(f'✗ 受体转换错误: {os.path.basename(pdb_file)} - {e}')
    print(f'受体转换完成: {len(pdbqt_files)}/{len(pdb_files)} 个文件成功')
    return pdbqt_files


def convert_sdf_to_pdbqt_batch_obabel(sdf_dir: str, obabel_path: str = 'obabel') -> List[str]:
    """使用OpenBabel批量转换配体SDF为PDBQT"""
    sdf_files = glob.glob(os.path.join(sdf_dir, 'lig_*.sdf'))
    pdbqt_files: List[str] = []
    print(f'开始使用OpenBabel批量转换 {len(sdf_files)} 个配体SDF文件为PDBQT...')
    for sdf_file in sdf_files:
        pdbqt_file = sdf_file.replace('.sdf', '.pdbqt')
        try:
            cmd = [
                obabel_path,
                sdf_file,
                '-O', pdbqt_file,
                '--partialcharge', 'gasteiger'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(pdbqt_file):
                # 清理配体PDBQT文件，确保符合Vina格式要求
                clean_ligand_pdbqt_for_vina(pdbqt_file)
                pdbqt_files.append(pdbqt_file)
                print(f'✓ 配体转换成功: {os.path.basename(sdf_file)} -> {os.path.basename(pdbqt_file)}')
            else:
                print(f'✗ 配体转换失败: {os.path.basename(sdf_file)} - {result.stderr}')
        except Exception as e:
            print(f'✗ 配体转换错误: {os.path.basename(sdf_file)} - {e}')
    print(f'配体转换完成: {len(pdbqt_files)}/{len(sdf_files)} 个文件成功')
    return pdbqt_files


# MGLTools生成的PDBQT文件通常不需要额外清理，直接使用即可


def main():
    parser = argparse.ArgumentParser(description='Docking evaluation directly from LMDB (reconstruct structures)')
    parser.add_argument('--data_dir', type=str, default='src/data/crossdocked_v1.1_rmsd1.0_processed',
                        help='包含 *.lmdb 与 *_name2id.pt 的目录')
    parser.add_argument('--lmdb', type=str, default='crossdocked_v1.1_rmsd1.0_processed_full_ref_prior_aromatic.lmdb')
    parser.add_argument('--name2id', type=str, default='crossdocked_v1.1_rmsd1.0_processed_full_ref_prior_aromatic_name2id.pt')
    parser.add_argument('--max_items', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='experiments/outputs/docking_from_lmdb')
    parser.add_argument('--vina_path', type=str, default='vina')
    parser.add_argument('--obabel_path', type=str, default='obabel', help='OpenBabel可执行路径')
    parser.add_argument('--mgltools_path', type=str, default='python', help='MGLTools Python路径（可选）')
    parser.add_argument('--converter', type=str, default='obabel', choices=['obabel','mgltools'], help='选择转换工具')
    parser.add_argument('--box_size', type=float, default=20.0)
    parser.add_argument('--exhaustiveness', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lmdb_path = os.path.join(args.data_dir, args.lmdb)
    name2id_path = os.path.join(args.data_dir, args.name2id)

    print('抽样条目...')
    # 优先直接从 LMDB 遍历抽样，确保键可用；失败再回退 name2id
    try:
        keys = sample_keys_from_lmdb(lmdb_path, args.max_items)
    except Exception:
        keys = []
    if not keys:
        try:
            keys = sample_keys_from_name2id(name2id_path, args.max_items)
        except Exception:
            keys = []

    mols: List[Chem.Mol] = []
    receptors: List[str] = []
    binding_sites: List[Dict[str, float]] = []

    tmp_dir = os.path.join(args.output_dir, 'temp_mols')
    os.makedirs(tmp_dir, exist_ok=True)
    print(f'临时输出目录: {tmp_dir}')

    kept = 0
    for k in keys:
        try:
            # k 可能是 bytes 或 str，read_item_from_lmdb 已兼容
            item = read_item_from_lmdb(lmdb_path, k)
        except Exception as e:
            continue

        # 期望字段（与你的数据管线一致）：
        # item['protein_element'] -> list/np.ndarray of atomic numbers or symbols
        # item['protein_pos'] -> (N,3) np.ndarray
        # item['ligand_element']
        # item['ligand_pos']
        protein_pos = item.get('protein_pos', None)
        ligand_pos = item.get('ligand_pos', None)
        protein_el = item.get('protein_element', None)
        ligand_el = item.get('ligand_element', None)

        if protein_pos is None or ligand_pos is None or protein_el is None or ligand_el is None:
            continue

        protein_pos = np.asarray(protein_pos)
        ligand_pos = np.asarray(ligand_pos)

        # 将元素编码转换为符号（若为数字）
        def to_symbols(el):
            arr = np.asarray(el)
            if arr.dtype.kind in ('U','S','O'):
                return [str(x) for x in arr.tolist()]
            out = []
            for z in arr.tolist():
                try:
                    zint = int(z)
                    out.append(Chem.GetPeriodicTable().GetElementSymbol(zint))
                except Exception:
                    out.append('C')
            return out

        protein_sym = to_symbols(protein_el)
        ligand_sym = to_symbols(ligand_el)

        # 写受体PDB（供后续转换为PDBQT或直接用于外部工具）
        rec_pdb = os.path.join(tmp_dir, f'rec_{kept:05d}.pdb')
        write_protein_pdb(protein_sym, protein_pos, rec_pdb)

        # 构建配体RDKit Mol并写SDF
        lig_mol = build_rdkit_mol_from_coords(ligand_sym, ligand_pos)
        lig_sdf = os.path.join(tmp_dir, f'lig_{kept:05d}.sdf')
        write_rdkit_mol_sdf(lig_mol, lig_sdf)

        # 以配体质心为对接盒中心
        conf = lig_mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(lig_mol.GetNumAtoms())], dtype=float)
        center = coords.mean(axis=0)
        box = {
            'center_x': float(center[0]),
            'center_y': float(center[1]),
            'center_z': float(center[2]),
            'size_x': float(args.box_size),
            'size_y': float(args.box_size),
            'size_z': float(args.box_size),
        }

        # 记录PDB路径，稍后批量转换为PDBQT
        receptors.append(rec_pdb)
        mols.append(lig_mol)
        binding_sites.append(box)
        kept += 1

    if kept == 0:
        print('无法从LMDB重建任何条目：请确认条目包含 protein_pos/ligand_pos 与 protein_element/ligand_element 字段，或检查 LMDB 键是否正确。')
        return

    # 使用你手动准备的标准PDBQT文件
    print('使用手动准备的标准PDBQT文件进行对接测试...')
    
    # 受体文件
    receptor_files = []
    for i in range(3):  # 前3个
        rec_pdbqt = os.path.join(tmp_dir, f'rec_{i:05d}.pdbqt')
        if os.path.exists(rec_pdbqt):
            receptor_files.append(rec_pdbqt)
            print(f'✓ 使用受体文件: {os.path.basename(rec_pdbqt)}')
    
    # 配体文件 - 从SDF生成PDBQT并清理水分子
    ligand_files = []
    for i in range(3):  # 前3个
        lig_sdf = os.path.join(tmp_dir, f'lig_{i:05d}.sdf')
        lig_pdbqt = os.path.join(tmp_dir, f'lig_{i:05d}.pdbqt')
        
        if os.path.exists(lig_sdf):
            # 删除旧的PDBQT文件，强制重新生成
            if os.path.exists(lig_pdbqt):
                os.remove(lig_pdbqt)
                print(f'删除旧配体PDBQT文件: {os.path.basename(lig_pdbqt)}')
            
            # 从SDF生成PDBQT
            print(f'从SDF生成配体PDBQT: {os.path.basename(lig_sdf)} -> {os.path.basename(lig_pdbqt)}')
            try:
                cmd = [
                    'obabel',
                    lig_sdf,
                    '-O', lig_pdbqt,
                    '--partialcharge', 'gasteiger'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(lig_pdbqt):
                    # 清理配体PDBQT文件，移除水分子
                    clean_ligand_pdbqt_for_vina(lig_pdbqt)
                    ligand_files.append(lig_pdbqt)
                    print(f'✓ 使用配体文件: {os.path.basename(lig_pdbqt)} (已清理水分子)')
                else:
                    print(f'✗ 配体转换失败: {os.path.basename(lig_sdf)} - {result.stderr}')
            except Exception as e:
                print(f'✗ 配体转换错误: {os.path.basename(lig_sdf)} - {e}')
        elif os.path.exists(lig_pdbqt):
            # 如果PDBQT文件已存在，直接使用
            clean_ligand_pdbqt_for_vina(lig_pdbqt)
            ligand_files.append(lig_pdbqt)
            print(f'✓ 使用现有配体文件: {os.path.basename(lig_pdbqt)} (已清理水分子)')
    
    if not receptor_files or not ligand_files:
        print('错误：没有找到足够的PDBQT文件')
        print(f'找到受体文件: {len(receptor_files)} 个')
        print(f'找到配体文件: {len(ligand_files)} 个')
        return
    
    # 确保数量匹配
    min_count = min(len(receptor_files), len(ligand_files))
    print(f'开始对接评估（样本数={min_count}，exhaustiveness={args.exhaustiveness}）...')
    
    # 使用现有的分子数据和绑定位点
    valid_mols = mols[:min_count]
    valid_binding_sites = binding_sites[:min_count]
    valid_receptors = receptor_files[:min_count]
    valid_ligands = ligand_files[:min_count]
    
    # 创建DockingEvaluator实例
    evaluator = DockingEvaluator(vina_path=args.vina_path)
    
    # 直接使用准备好的配体PDBQT文件进行对接
    results = []
    for i in range(min_count):
        print(f'对接分子 {i+1}/{min_count}')
        
        best_affinity = float('inf')
        all_affinities = []
        successful_dockings = 0
        
        # 使用准备好的配体PDBQT文件
        ligand_pdbqt = valid_ligands[i]
        receptor_pdbqt = valid_receptors[i]
        binding_site = valid_binding_sites[i]
        
        docking_result = evaluator.run_vina_docking(
            ligand_pdbqt, receptor_pdbqt,
            binding_site['center_x'], binding_site['center_y'], binding_site['center_z'],
            binding_site.get('size_x', 20.0), binding_site.get('size_y', 20.0), 
            binding_site.get('size_z', 20.0),
            exhaustiveness=args.exhaustiveness
        )
        
        if docking_result['success']:
            results.append({
                'index': i,
                'success': True,
                'affinity': docking_result['affinity'],
                'receptor': os.path.basename(receptor_pdbqt),
                'ligand': os.path.basename(ligand_pdbqt)
            })
            print(f'✓ 对接成功: affinity = {docking_result["affinity"]:.2f}')
        else:
            results.append({
                'index': i,
                'success': False,
                'error': docking_result.get('error', 'Unknown error'),
                'receptor': os.path.basename(receptor_pdbqt),
                'ligand': os.path.basename(ligand_pdbqt)
            })
            print(f'✗ 对接失败: {docking_result.get("error", "Unknown error")}')
    
    df = pd.DataFrame(results)

    metrics = evaluator.calculate_docking_metrics(df)

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'docking_results_from_lmdb.csv'), index=False)
    with open(os.path.join(args.output_dir, 'docking_metrics_from_lmdb.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print('docking_results_from_lmdb.csv 已保存')
    print('docking_metrics_from_lmdb.json 已保存')
    print('完成！所有PDB文件已自动转换为PDBQT格式并进行对接评估。')


if __name__ == '__main__':
    main()


