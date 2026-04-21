"""
Main entry point for multi-agent PPO/MBPO training.

Usage:
    # Discrete (default) — unchanged from original
    python main.py --env simple_adversary

    # Continuous action space
    python main.py --env simple_adversary --action_space continuous

    # Both teams MBPO, continuous
    python main.py --env simple_adversary --good_algo mbpo --adv_algo mbpo --action_space continuous

    # Custom hyperparameters
    python main.py --env simple_tag --updates 500 --rollout 4096 --action_space continuous

    # Multiple runs
    python main.py --env simple_adversary --runs 5 --action_space continuous
"""

import argparse
import time
import os
from config import Config
from trainer import MultiAgentTrainer
from utils import (
    create_env,
    setup_logging,
    restore_logging,
    print_config_summary,
    format_time,
    save_training_history,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent PPO/MBPO Training on PettingZoo MPE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default discrete action space
  python main.py --env simple_adversary

  # Continuous action space
  python main.py --env simple_adversary --action_space continuous

  # Both teams MBPO, continuous
  python main.py --env simple_tag --good_algo mbpo --adv_algo mbpo --action_space continuous

  # Multiple runs
  python main.py --env simple_adversary --runs 5

Available environments:
  simple_adversary, simple_crypto, simple_push, simple_reference,
  simple_speaker_listener, simple_spread, simple_tag, simple_world_comm
        """
    )

    # Environment
    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=[
            'simple_adversary', 'simple_crypto', 'simple_push',
            'simple_reference', 'simple_speaker_listener', 'simple_spread',
            'simple_tag', 'simple_world_comm',
        ],
        help='PettingZoo MPE environment name',
    )

    # Action space toggle  ← NEW
    parser.add_argument(
        '--action_space',
        type=str,
        default='discrete',
        choices=['discrete', 'continuous'],
        help='Action space type: discrete (default) or continuous',
    )

    # Algorithm selection
    parser.add_argument('--good_algo', type=str, default='mbpo', choices=['ppo', 'mbpo'],
                        help='Algorithm for good agents (default: mbpo)')
    parser.add_argument('--adv_algo',  type=str, default='ppo',  choices=['ppo', 'mbpo'],
                        help='Algorithm for adversary agents (default: ppo)')

    # Training settings
    parser.add_argument('--updates',      type=int,   default=250,  help='Number of training updates')
    parser.add_argument('--rollout',      type=int,   default=2048, help='Rollout steps per update')
    parser.add_argument('--runs',         type=int,   default=1,    help='Number of training runs')

    # Device
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help='Device to use (default: auto-detect)')

    # Logging
    parser.add_argument('--log_file',     type=str, default=None,
                        help='File to log output to')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N updates (default: 10)')

    # Saving / Loading
    parser.add_argument('--save_dir',     type=str,            default=None,
                        help='Directory to save trained agents')
    parser.add_argument('--save_history', action='store_true',
                        help='Save training history to file')

    # Evaluation
    parser.add_argument('--eval',          action='store_true',
                        help='Run evaluation after training')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')

    # Hyperparameter overrides
    parser.add_argument('--lr_actor',  type=float, default=None)
    parser.add_argument('--lr_critic', type=float, default=None)
    parser.add_argument('--lr_model',  type=float, default=None)
    parser.add_argument('--gamma',     type=float, default=None)
    parser.add_argument('--lam',       type=float, default=None)

    # Continuous-specific overrides
    parser.add_argument('--log_std_init', type=float, default=None,
                        help='Initial log_std for continuous policy (default: -0.5)')
    parser.add_argument('--log_std_min',  type=float, default=None,
                        help='Minimum log_std clamp (default: -3.0)')
    parser.add_argument('--log_std_max',  type=float, default=None,
                        help='Maximum log_std clamp (default:  1.0)')

    return parser.parse_args()


def create_config(args) -> Config:
    cfg = Config()

    # Environment and algorithms
    cfg.env_name          = f"{args.env}_v3"
    cfg.good_agent_algo   = args.good_algo
    cfg.adversary_algo    = args.adv_algo
    cfg.action_space_type = args.action_space   # ← NEW

    # Training settings
    cfg.max_updates   = args.updates
    cfg.rollout_steps = args.rollout
    cfg.log_interval  = args.log_interval

    # Device
    if args.device is not None:
        cfg.device = args.device

    # Standard hyperparameter overrides
    if args.lr_actor  is not None: cfg.lr_actor  = args.lr_actor
    if args.lr_critic is not None: cfg.lr_critic = args.lr_critic
    if args.lr_model  is not None: cfg.lr_model  = args.lr_model
    if args.gamma     is not None: cfg.gamma     = args.gamma
    if args.lam       is not None: cfg.lam       = args.lam

    # Continuous-specific overrides
    if args.log_std_init is not None: cfg.log_std_init = args.log_std_init
    if args.log_std_min  is not None: cfg.log_std_min  = args.log_std_min
    if args.log_std_max  is not None: cfg.log_std_max  = args.log_std_max

    return cfg


def run_single_training(args, cfg: Config, run_id: int = 1):
    print("\n" + "=" * 70)
    print(f"STARTING RUN {run_id}")
    print("=" * 70 + "\n")

    print(f"Creating environment: {cfg.env_name} (action_space={cfg.action_space_type})")
    env = create_env(args.env, continuous_actions=cfg.is_continuous)   # ← NEW flag

    print_config_summary(cfg)

    trainer = MultiAgentTrainer(env, cfg)

    start_time = time.time()
    history    = trainer.train(cfg.max_updates)
    training_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Training completed in {format_time(training_time)}")
    print(f"{'='*70}\n")

    if args.eval:
        trainer.evaluate(args.eval_episodes)

    if args.save_dir is not None:
        save_path = os.path.join(args.save_dir, f"run_{run_id}")
        trainer.save_agents(save_path)

    if args.save_history:
        history_file = (
            f"{args.env}_{cfg.action_space_type}_"
            f"{cfg.good_agent_algo}_vs_{cfg.adversary_algo}_run{run_id}_history.npz"
        )
        save_training_history(history, history_file)

    trainer.close()

    print(f"\n{'='*70}")
    print(f"RUN {run_id} COMPLETE")
    print(f"{'='*70}\n")

    return history


def main():
    args = parse_args()

    if args.log_file is None:
        args.log_file = (
            f"{args.env}_{args.action_space}_"
            f"{args.good_algo}_vs_{args.adv_algo}_output.txt"
        )

    original_stdout, log_file = setup_logging(args.log_file)

    try:
        cfg = create_config(args)

        all_histories = []
        for run in range(1, args.runs + 1):
            history = run_single_training(args, cfg, run_id=run)
            all_histories.append(history)

        if args.runs > 1:
            print("\n" + "=" * 70)
            print("SUMMARY OF ALL RUNS")
            print("=" * 70)
            print(f"Total runs completed: {args.runs}")
            print(f"Environment:    {args.env}")
            print(f"Action Space:   {args.action_space}")
            print(f"Good agents:    {args.good_algo.upper()}")
            print(f"Adversary agents: {args.adv_algo.upper()}")
            print("=" * 70 + "\n")

        print(f"All training completed!")
        print(f"Output logged to: {args.log_file}")

    finally:
        restore_logging(original_stdout, log_file)


if __name__ == "__main__":
    main()