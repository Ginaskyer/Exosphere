import argparse
from train import train_main, test_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training mode')
    parser.add_argument('--test', action='store_true', help='Run testing mode')
    args = parser.parse_args()

    if args.train:
        train_main()
    elif args.test:
        test_main()
    else:
        print("Please specify --train or --test")