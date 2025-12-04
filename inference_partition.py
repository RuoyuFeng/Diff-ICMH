from typing import List, Tuple, Optional
import os
import copy
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.ddim_sampler import DDIMSampler
from model.diffeic import DiffEIC
from utils.image import pad
from utils.metrics import compute_psnr, compute_ssim, LPIPS
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts


@torch.no_grad()
def process(
    model: DiffEIC,
    imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    stream_path: str,
    anchor_prior_strength: Optional[float] = None
) -> Tuple[List[np.ndarray], float]:
    """
    Apply DiffEIC model on a list of images.
    
    Args:
        model (DiffEIC): Model.
        imgs (List[np.ndarray]): A list of images (HWC, RGB, range in [0, 255])
        sampler (str): Sampler name.
        steps (int): Sampling steps.
        stream_path (str): Savedir of bitstream
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        bpp
    """
    n_samples = len(imgs)
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    height, width = control.size(-2), control.size(-1)
    if model.preprocess_tag_model.enabled:
        c_tag_ids, _ = model.preprocess_tag_model(control, return_ids=True)
        c_tag_ids = [each for each in c_tag_ids[0].reshape(-1)]

    if model.preprocess_tag_model.enabled:
        bpp = model.apply_condition_compress(control, stream_path, height, width, tag_ids=c_tag_ids)
    else:
        bpp = model.apply_condition_compress(control, stream_path, height, width)
    if model.preprocess_tag_model.enabled:
        c_latent, c_tag_ids = model.apply_condition_decompress(stream_path, dec_tag_ids=True)
        # Convert decoded tag IDs to tensor format for index2tag
        # c_tag_ids_tensor = torch.tensor(c_tag_ids).reshape(1, -1)  # Shape: [1, num_tags]
        c_tag_ids_np = np.array(c_tag_ids).reshape(-1, 1)  # Shape: [1, num_tags]
        c_tag_rec = model.preprocess_tag_model.model.index2tag([c_tag_ids_np])[0]
        c_tag_rec = [tag.replace(' |', ',') for tag in c_tag_rec]
    else:
        c_latent = model.apply_condition_decompress(stream_path)

    cond = {
        "c_latent": [c_latent],
        "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]
    }
    unconditional_guidance_scale = 1.0
    unconditional_conditioning = None
    # if model.preprocess_semantic_model.enabled:
    #     import pdb; pdb.set_trace()
    #     c_semantic, bits_sem = model.preprocess_semantic_model(control)  # control: [0, 1]
    #     cond["c_crossattn"] = [c_semantic] * n_samples
    #     unconditional_guidance_scale = model.c_cfg_scale
    #     unconditional_conditioning = copy.deepcopy(cond)
    #     unconditional_conditioning["c_crossattn"] = [torch.zeros_like(cond["c_crossattn"][0])] * n_samples
    if model.preprocess_semantic_model.enabled or model.preprocess_tag_model.enabled:
        unconditional_guidance_scale = model.c_cfg_scale
        unconditional_conditioning = copy.deepcopy(cond)
        if model.preprocess_semantic_model.enabled:
            c_semantic, bits_sem = model.preprocess_semantic_model(control)  # control: [0, 1]
            # cond["c_crossattn"] = [c_semantic] * n_samples
            cond_crossattn = c_semantic
            n_tokens = 256 // (model.preprocess_semantic_model.feature_postprocessor.postprocess_downscale**2)
            uncond_semantic = torch.zeros(n_samples, n_tokens, cond["c_crossattn"][0].shape[-1]).to(model.device)
            uncond_crossattn = uncond_semantic
        if model.preprocess_tag_model.enabled:
            # c_tag, bits_tag = model.preprocess_tag_model(control)
            c_tag = c_tag_rec
            cond_tag = model.cond_stage_model.encode(c_tag)
            uncond_tag = model.cond_stage_model.encode([''] * n_samples)
            if model.preprocess_semantic_model.enabled:
                cond_crossattn = torch.cat([cond_crossattn, cond_tag], 1)
                uncond_crossattn = torch.cat([uncond_crossattn, uncond_tag], 1)
            else:
                cond_crossattn = cond_tag
                uncond_crossattn = uncond_tag
        cond["c_crossattn"] = [cond_crossattn]
        unconditional_conditioning["c_crossattn"] = [uncond_crossattn]
    
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if anchor_prior_strength is not None:
        x_T = x_T + anchor_prior_strength * cond['c_latent'][0]
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            # unconditional_guidance_scale=1.0,
            # unconditional_conditioning=None,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            cond_fn=None, x_T=x_T
        )
    else:
        sampler: DDIMSampler
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            x_T=x_T, eta=0
        )
    
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    
    return preds, bpp


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt_sd", default='./checkpoints/sd2p1/v2-1_512-ema-pruned.ckpt', type=str, help="checkpoint path of stable diffusion")
    parser.add_argument("--ckpt_lc", default='path to checkpoint file of lfgcm and control module', type=str, help="checkpoint path of lfgcm and control module")
    parser.add_argument("--config", default='configs/model/diffeic.yaml', type=str, help="model config path")
    
    parser.add_argument("--input", type=str, default='path to input images')
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=50, type=int)
    
    parser.add_argument("--output", type=str, default='results/')
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--anchor_prior_strength', type=float, default=None)
    
    parser.add_argument('overrides', nargs='*', help='override model config keys')

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()

    # model: DiffEIC = instantiate_from_config(OmegaConf.load(args.config))
    model_config = OmegaConf.load(args.config)
    # process overrides
    overrides = args.overrides
    if overrides:
        model_config = OmegaConf.merge(model_config, OmegaConf.from_dotlist(overrides))
        print('Merged model config')
        print(OmegaConf.to_yaml(model_config))
    model: DiffEIC = instantiate_from_config(model_config)
    ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")['state_dict']
    ckpt_lc = torch.load(args.ckpt_lc, map_location="cpu")['state_dict']
    ckpt_sd.update(ckpt_lc)
    msg = load_state_dict(model, ckpt_sd, strict=False)
    print(f"Messgae of load state dict: {msg}")
    # update preprocess model
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)
    anchor_prior_strength = args.anchor_prior_strength

    bpps = []
    psnrs, ssims, lpips_values = [], [], []
    file_metrics = []
    
    assert os.path.isdir(args.input)

    # Intialize the LPIPS model
    lpips = LPIPS('alex').to(args.device)
    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    for file_path in list_image_files(args.input, follow_links=True):
        img = Image.open(file_path).convert("RGB")
        x = pad(np.array(img), scale=64)
        
        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, 'data')
        save_path = os.path.join(parent_path, f"{stem}.png")
        stream_path = os.path.join(stream_parent_path, f"{stem}")

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)
        
        preds, bpp = process(
            model, [x], steps=args.steps, sampler=args.sampler,
            stream_path=stream_path,
            anchor_prior_strength=anchor_prior_strength
        )
        pred = preds[0]

        # calculate bpp and save to list
        bpps.append(bpp)
        relative_file_path = os.path.relpath(file_path, args.input)

        # remove padding
        x_tmp = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255
        xhat_tmp = torch.tensor(pred).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255
        psnr = compute_psnr(x_tmp, xhat_tmp)
        ssim = compute_ssim(x_tmp, xhat_tmp)    # seems wrong results? better to use the [ pyiqa ] package
        lpips_value = lpips(x_tmp, xhat_tmp, normalize=True).mean()
        psnrs.append(psnr.item())
        ssims.append(ssim.item())
        lpips_values.append(lpips_value.item())
        file_metrics.append({
            'relative_path': relative_file_path,
            'bpp': bpp,
            'psnr': psnr.item(),
            'ssim': ssim.item(),
            'lpips': lpips_value.item()
        })
        
        pred = pred[:img.height, :img.width, :]
        Image.fromarray(pred).save(save_path)
        print(f"save to {save_path}, bpp {bpp}")

    avg_bpp = sum(bpps) / len(bpps)
    avg_psnr = sum(psnrs) / len(psnrs)
    avg_ssim = sum(ssims) / len(ssims)
    avg_lpips = sum(lpips_values) / len(lpips_values)
    print(f'avg bpp: {avg_bpp:.4f}')
    print(f'avg psnr: {avg_psnr:.4f}')
    print(f'avg ssim: {avg_ssim:.4f}')
    print(f'avg lpips: {avg_lpips:.4f}')

    # write bpp to file with individual file bpp and average bpp
    bpp_file_path = os.path.join(args.output, 'bpp.txt')
    with open(bpp_file_path, 'w') as f:
        f.write(f'filename: bpp / psnr / ssim / lpips\n')
        for result in file_metrics:
            f.write(f"{result['relative_path']}: {result['bpp']:.4f} / {result['psnr']:.4f} / {result['ssim']:.4f} / {result['lpips']:.4f}\n")
        f.write(f'\navg: {avg_bpp:.4f} / {avg_psnr:.4f} / {avg_ssim:.4f} / {avg_lpips:.4f}\n')


if __name__ == "__main__":
    main()
