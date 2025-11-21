import argparse
import datetime
import os
import random
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mi_identifiability.circuit import find_circuits
from mi_identifiability.mappings import find_minimal_mappings
from mi_identifiability.logic_gates import ALL_LOGIC_GATES, generate_noisy_multi_gate_data, get_formula_dataset
from mi_identifiability.neural_model import MLP
from mi_identifiability.utils import setup_logging, set_seeds


def run_experiment(logger, output_dir, run_dir, args):
    data = []

    if args.resume_from is not None:
        # Read output from previous run
        prev_run_dir = output_dir / args.resume_from
        in_file = prev_run_dir / 'data_tmp.csv'
        try:
            run_dir = prev_run_dir
            data = pd.read_csv(in_file).to_dict('records')
            in_file.rename(prev_run_dir / 'data_tmp.bak')
        except FileNotFoundError:
            logger.error(f'Previous run data {args.resume_from} not found, starting from scratch')

    all_logic_gates = [ALL_LOGIC_GATES[gate] for gate in args.target_logic_gates]
    formula_dataset = {depth: get_formula_dataset(all_logic_gates, max_depth=depth + 1) for depth in args.depth}

    for n_gates, seed_offset in product(args.n_gates, range(args.n_experiments)):
        gates = random.sample(all_logic_gates, k=n_gates)
        n_inputs = gates[0].n_inputs

        if n_gates > 1:
            x_tt, y_tt = generate_noisy_multi_gate_data(gates, n_repeats=args.n_repeats, noise_std=args.noise_std,
                                                        device=args.device)
        else:
            x_tt, y_tt = gates[0].generate_noisy_data(n_repeats=args.n_repeats, noise_std=args.noise_std,
                                                      device=args.device)

        print(f'Iteration # {seed_offset}')
        weights = None
        if args.skewed_distribution:
            weights = [random.random() for _ in range(2 ** n_inputs)]

        if n_gates > 1:
            x, y = generate_noisy_multi_gate_data(gates, n_repeats=args.n_samples_train, weights=weights,
                                                  noise_std=args.noise_std, device=args.device)
            x_val, y_val = generate_noisy_multi_gate_data(gates, n_repeats=args.n_samples_val, weights=weights,
                                                          noise_std=args.noise_std, device=args.device)
        else:
            x, y = gates[0].generate_noisy_data(n_repeats=args.n_samples_train, weights=weights,
                                                noise_std=args.noise_std, device=args.device)
            x_val, y_val = gates[0].generate_noisy_data(n_repeats=args.n_samples_val, weights=weights,
                                                        noise_std=args.noise_std, device=args.device)

        for k, depth, loss_target, lr in product(args.size, args.depth, args.loss_target, args.learning_rate):
            set_seeds(args.seed + seed_offset)

            skip_exp = False

            for e in data:
                if (e['gates'] == ' '.join(lg.name for lg in gates) and e['n_gates'] >= n_gates and e['sizes'] >= k and
                        e['seed_offset'] >= seed_offset and e['loss_target'] >= loss_target and e['learning_rate'] >= lr
                        and e['depth'] >= depth):
                    print('Skipping already recorded run')
                    skip_exp = True
                    break

            if skip_exp:
                continue

            layer_sizes = [2] + [k] * depth + [n_gates]
            
            # Create model with regularization parameters
            model = MLP(
                hidden_sizes=layer_sizes[1:-1], 
                input_size=n_inputs, 
                output_size=n_gates, 
                device=args.device,
                dropout_rate=args.dropout_rate  # NEW
            )

            # Train with regularization
            avg_loss = model.do_train(
                x=x,
                y=y,
                x_val=x_val,
                y_val=y_val,
                batch_size=args.batch_size,
                learning_rate=lr,
                epochs=args.epochs,
                loss_target=loss_target,
                val_frequency=args.val_frequency,
                early_stopping_steps=args.early_stopping_steps,
                logger=logger if args.verbose else None,
                l1_lambda=args.l1_lambda,
                l2_lambda=args.l2_lambda
            )

            val_acc = model.do_eval(x_val, y_val)

            if val_acc < 0.99 or avg_loss > loss_target:
                logger.info(f"No convergence - k={k}, seed={seed_offset}, loss={avg_loss}/{loss_target}, acc={val_acc}")
                continue

            # Use min_sparsity from args, defaults to 0.0 for full circuit evaluation
            # Setting to 0 allows us to see the full effect of regularization on circuit counts
            min_sparsity = args.min_sparsity if args.min_sparsity is not None else 0.0

            n_circuits = []

            submodels = model.separate_into_k_mlps()
            for i, submodel in enumerate(submodels):
                top_circuits, sparsities, _ = find_circuits(
                    submodel,
                    x_val,
                    y_val[..., i].unsqueeze(1),
                    accuracy_threshold=args.accuracy_threshold,
                    min_sparsity=min_sparsity,
                    use_gpu_batching=args.use_gpu_batching,
                    batch_size=args.gpu_batch_size
                )

                n_circuits.append(len(top_circuits))

                if args.max_circuits is not None and args.max_circuits < len(top_circuits):
                    sample_top_circuits = [c for c, _ in sorted(zip(top_circuits, sparsities),
                                                                key=lambda t: t[1], reverse=True)[:args.max_circuits]]
                else:
                    sample_top_circuits = top_circuits

            # Add regularization info to saved data
            data.append({
                'gates': ' '.join(lg.name for lg in gates),
                'n_gates': n_gates,
                'sizes': k,
                'seed_offset': seed_offset,
                'perfect_circuits': n_circuits,
                'loss_target': loss_target,
                'learning_rate': lr,
                'depth': depth,
                'weights': weights,
                'l1_lambda': args.l1_lambda,  # NEW
                'l2_lambda': args.l2_lambda,  # NEW
                'dropout_rate': args.dropout_rate  # NEW
            })
            pd.DataFrame(data).to_csv(os.path.join(run_dir, 'data_tmp.csv'), index=False)

    return run_dir, pd.DataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Starting random seed')
    parser.add_argument('--size', type=int, nargs='+', default=[3],
                        help='Size of the hidden layers of the neural networks')
    parser.add_argument('--depth', type=int, nargs='+', default=[2],
                        help='Number of hidden layers in the neural networks')
    parser.add_argument('--n-repeats', type=int, default=10,
                        help='Number of times to repeat the data for circuit search')
    parser.add_argument('--n-experiments', type=int, default=100,
                        help='Runs each experiment n times with different seeds')
    parser.add_argument('--noise-std', type=float, default=0.1, help='Standard deviation of the Gaussian noise')
    parser.add_argument('--n-samples-train', type=int, default=1000, help='Number of samples for training')
    parser.add_argument('--n-samples-val', type=int, default=50, help='Number of samples for validation')
    parser.add_argument('--n-gates', type=int, nargs='+', default=[1],
                        help='Number of logic gates used as target functions')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, nargs='+', default=[0.001],
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--max-circuits', type=int, default=None, help='Maximum number of circuits used for grounding')
    parser.add_argument('--min-sparsity', type=float, default=0.0, help='Minimum sparsity for circuit search')
    parser.add_argument('--loss-target', type=float, nargs='+', default=[0.01],
                        help='Target loss value for training')
    parser.add_argument('--skewed-distribution', action='store_true',
                        help='Whether to use a skewed distribution for training data')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to store the data on')
    parser.add_argument('--target-logic-gates', type=str, nargs='+',
                        default=['AND', 'OR', 'IMP', 'NAND', 'NOR', 'NIMP', 'XOR'], choices=ALL_LOGIC_GATES.keys(),
                        help='The allowed target logic gates')
    parser.add_argument('--accuracy-threshold', type=float, default=0.99,
                        help='The accuracy threshold for circuit search')
    parser.add_argument('--val-frequency', type=int, default=10, help='Frequency of validation during training')
    parser.add_argument('--early-stopping-steps', type=int, default=30,
                        help='Number of epochs without improvement before stopping training (default: 30)')
    parser.add_argument('--verbose', action='store_true', help='Whether to display extra information')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from a previous run')

    # NEW: Regularization arguments
    parser.add_argument('--l1-lambda', type=float, default=0.0,
                        help='L1 regularization coefficient (default: 0.0)')
    parser.add_argument('--l2-lambda', type=float, default=0.0,
                        help='L2 regularization coefficient (default: 0.0)')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='Dropout rate, 0.0 to 1.0 (default: 0.0)')

    # Circuit discovery optimization arguments
    parser.add_argument('--use-gpu-batching', action='store_true',
                        help='Use GPU batching for faster circuit discovery')
    parser.add_argument('--gpu-batch-size', type=int, default=128,
                        help='Batch size for GPU circuit evaluation (default: 128)')

    args = parser.parse_args()
    output_dir = Path('logs')

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"

    if args.resume_from is None:
        logger = setup_logging(run_dir, "output.log")
    else:
        logger = setup_logging(output_dir / args.resume_from, "output.log")

    logger.info("Configuration in use:")
    logger.info(args)

    seed = args.seed
    set_seeds(seed)
    logger.info(f"Setting the seeds: {seed}")

    run_dir, df_res = run_experiment(logger, output_dir, run_dir, args)

    output_filepath = run_dir / "df_out.csv"
    df_res.to_csv(output_filepath)
    logger.info(f"Results were saved at {output_filepath}")

    print(df_res)
