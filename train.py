# Original code is in train-bkp.py

import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities.cli import LightningCLI

from utils.common import instantiate_from_config, load_state_dict


def set_rank_specific_seed(base_seed: int):
    """
    Set rank-specific random seed.

    Args:
        base_seed (int): Base random seed.
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0  # rank is 0 for single-machine training
    rank_seed = base_seed + rank
    pl.seed_everything(rank_seed, workers=True)
    print(f"Rank {rank} random seed set to: {rank_seed}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/train_diffeic.yaml')
    parser.add_argument('--resume_codec', action='store_true', help='resume weights of codec only')
    parser.add_argument('overrides', nargs='*', help='override model config keys')
    args = parser.parse_args()
    
    # Load the config file.
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
    
    # Merge the config with the overrides.
    overrides = args.overrides
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
        print('Merged config')
        print(OmegaConf.to_yaml(config))

    # Load the data module and model.
    data_module = instantiate_from_config(config.data)

    # Merge the model config with the overrides.
    # overrides_model = [o.replace('model.', '') for o in overrides if o.startswith('model.')]
    overrides_model = [o.replace('model.params.', 'params.') for o in overrides if o.startswith('model.params.')]
    config_model = OmegaConf.load(config.model.config)
    if overrides_model:
        config_model = OmegaConf.merge(config_model, OmegaConf.from_dotlist(overrides_model))
        print('Merged model config')
        print(OmegaConf.to_yaml(config_model))
    model = instantiate_from_config(config_model)
    
    # Write the final config to the saving directory.
    save_dir = config.lightning.trainer.get("default_root_dir")
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    if save_dir:
        with open(f"{save_dir}/config.yaml", "w") as fp:
            OmegaConf.save(config, fp)
        print(f"Saved config to {save_dir}/config.yaml")
        with open(f"{save_dir}/config_model.yaml", "w") as fp:
            OmegaConf.save(config_model, fp)
        print(f"Saved config to {save_dir}/config_model.yaml")

    if config.model.get("resume"):
        # load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=True)
        if args.resume_codec:
            ckpt = torch.load(config.model.resume, map_location="cpu")
            # filter out keys that are not of the codec
            ckpt['state_dict'] = {k: v for k, v in ckpt['state_dict'].items() if 'preprocess_model' in k}
            msg = load_state_dict(model, ckpt, strict=False)
        else:
            msg = load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=False)
        print(f"Loaded model state dict from {config.model.resume}, with message: {msg}")
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
