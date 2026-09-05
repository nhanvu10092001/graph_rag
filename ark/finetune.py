from fine_tuning.config import FineTuningConfig
from fine_tuning.graph_fine_tuning import run_fine_tuning
import argparse
import yaml

parser = argparse.ArgumentParser(description="Fine-tune a model on graph trajectories")
parser.add_argument("--config", type=str, default="fine_tuning/params.yaml", help="Path to YAML config file")
parser.add_argument("--graph_name", type=str, default=None)
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--train_queries_limit", type=int, default=None)
parser.add_argument("--val_queries_limit", type=int, default=None)
parser.add_argument("--rejection_sampling", type=str, default=None)
args = parser.parse_args()

with open(args.config, "r") as f:
    dict_config = yaml.safe_load(f)

# Override with command-line arguments if provided
args_config = {}
if args.graph_name is not None:
    args_config["graph_name"] = args.graph_name
if args.model_name is not None:
    args_config["model_name"] = args.model_name
if args.train_queries_limit is not None:
    args_config["train_queries_limit"] = args.train_queries_limit
if args.val_queries_limit is not None:
    args_config["val_queries_limit"] = args.val_queries_limit
if args.rejection_sampling is not None:
    args_config["rejection_sampling"] = args.rejection_sampling.lower() in ("true", "1", "yes")


# Merge configs (args override YAML)
dict_config = {**dict_config, **args_config}
config = FineTuningConfig.model_validate(dict_config)
run_fine_tuning(config)
