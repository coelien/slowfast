import sys
sys.path.append('./slowfast')
sys.path.append('./slowfast/datasets')
print(sys.path)
from slowfast.utils import parse_args, load_config
from train_net import train


def main():
    args = parse_args()
    print(args)
    print("config files: {}".format(args.cfg_files))
    # assume only one config file is provided
    config_file_path = args.cfg_files[0]
    cfg = load_config(args, config_file_path)
    train(cfg)


if __name__ == "__main__":
    main()
