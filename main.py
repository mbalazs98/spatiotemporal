from datasets import RawHD_dataloaders
from best_config_RawHD import Config
from snn_delays import SnnDelays
import torch
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")

config = Config()

model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()


print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")
print(f"===> Model size = {utils.count_parameters(model)}\n\n")


if config.dataset == 'rawhd':
    train_loader, valid_loader = RawHD_dataloaders(config)
    test_loader = None
else:
    raise Exception(f'dataset {config.dataset} not implemented')


model.train_model(train_loader, valid_loader, test_loader, device)
