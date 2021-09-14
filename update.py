import argparse
from src.getDataFromCromwell import update

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', dest='config', type=str,
                        default="config.json")

    args = parser.parse_args()

    if args.alltime and args.days != 0:
        print("Warning\nIgnoring the number of days argument\nGetting all data!")

    print(update(config=args.config))
