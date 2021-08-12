import argparse


from webApp.services.getDataFromCromwell import update

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', dest='config', type=str,
                        default="config.json")
    parser.add_argument('--all', dest='alltime', type=bool, default=False,
                        action='store_true')

    args = parser.parse_args()

    print(update(config=args.config, alltime=args.alltime))
