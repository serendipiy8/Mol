import os
import sys
import argparse
import torch


def add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


add_src_to_path()

from src.training.runner import run_train, run_sample_and_eval


def main():
    parser = argparse.ArgumentParser(description='Main Runner')
    parser.add_argument('--mode', type=str, default='train_and_sample', choices=['train', 'sample', 'train_and_sample'])
    parser.add_argument('--dataset_path', type=str, default='src/data/crossdocked_v1.1_rmsd1.0_processed')
    parser.add_argument('--split_file', type=str, default='src/data/split_by_name.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='Model/experiments/outputs/main')
    # model
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--use_cross_mp', action='store_true', default=True)
    parser.add_argument('--use_protein_context', action='store_true', default=True)
    parser.add_argument('--context_dropout', type=float, default=0.0)
    parser.add_argument('--cross_radius', type=float, default=6.0)
    parser.add_argument('--cross_topk', type=int, default=24)
    parser.add_argument('--bond_classes', type=int, default=5)
    parser.add_argument('--atom_type_classes', type=int, default=10)
    parser.add_argument('--protein_encoder_se3', action='store_true', default=False)
    # trainer
    parser.add_argument('--train_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--aggregate_all_t', action='store_true', default=False)
    parser.add_argument('--lambda_eps', type=float, default=0.0)
    parser.add_argument('--lambda_tau_smooth', type=float, default=0.0)
    parser.add_argument('--lambda_tau_rank', type=float, default=0.0)
    parser.add_argument('--lambda_atom_type', type=float, default=0.0)
    # sampling
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--de_novo', action='store_true', default=False)
    parser.add_argument('--de_novo_num_atoms', type=int, default=16)
    parser.add_argument('--de_novo_sigma', type=float, default=4.0)
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'sample':
        run_sample_and_eval(args)
    else:
        run_train(args)
        run_sample_and_eval(args)


if __name__ == '__main__':
    main()


