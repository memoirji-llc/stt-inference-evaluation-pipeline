import yaml
import wandb

with open("configs/inference.yaml") as f:
    config = yaml.safe_load(f)

wandb.init(project="amia-stt-testing", config=config)
print("placeholder statement")
wandb.log({"placeholder": 1})