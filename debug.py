import yaml, sys
cfg_path = "configs/newconfig.yaml"   # confirm this is correct

cfg = yaml.safe_load(open(cfg_path))
print("Using config file:", cfg_path, file=sys.stderr)
print("data block:", cfg.get("data"), file=sys.stderr)
