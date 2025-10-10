# cli.py
import argparse
from .viewer import load_outputs, plot_nav

def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_folder")
    p.add_argument("--save", default=None)
    args = p.parse_args()
    data = load_outputs(args.exp_folder)
    plot_nav(data["nav"], save_path=args.save)

if __name__ == "__main__":
    main()
