import argparse
from src.train import train
from config.config_loader import CONFIG

def main():
    """
    This script allows performing different tasks like training, evaluation, or prediction.
    """
    parser = argparse.ArgumentParser(description="Entry point for various tasks.")
    parser.add_argument('--task', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Task to perform: train, evaluate, or predict.')

    args = parser.parse_args()

    if args.task == 'train':
        # Train the model
        train(CONFIG)
    elif args.task == 'evaluate':
        # Call an evaluate function (to be implemented)
        print("Evaluation task is not implemented yet.")
    elif args.task == 'predict':
        # Call a predict function (to be implemented)
        print("Prediction task is not implemented yet.")

if __name__ == "__main__":
    main()
