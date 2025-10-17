from pathlib import Path
import torch
from .vq import RVQVAE, LengthEstimator
from .mask_transformer import MaskTransformer, ResidualTransformer

def load_vq_model(vq_opt, ckpt="net_best_fid.tar", dim_pose=251, device='cuda'):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt_path = Path(vq_opt.checkpoints_dir) / vq_opt.dataset_name / vq_opt.name / 'model' / ckpt
    
    if not ckpt_path.exists():
        ckpt_path = Path(vq_opt.checkpoints_dir) / vq_opt.dataset_name / vq_opt.name / 'model' / 'base.tar'
        print(f'WARNING: Checkpoint {ckpt} not found, using base.tar instead: {ckpt_path}')
    
    ckpt = torch.load(ckpt_path,
                            map_location=device)
    print(f'Loading VQ Model {ckpt_path}!')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    return vq_model, vq_opt

def load_trans_model(model_opt, ckpt="net_best_fid.tar", device='cuda', clip_version='ViT-B/32'):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt_path = Path(model_opt.checkpoints_dir) / model_opt.dataset_name / model_opt.name / 'model' / ckpt
    
    if not ckpt_path.exists():
        ckpt_path = Path(model_opt.checkpoints_dir) / model_opt.dataset_name / model_opt.name / 'model' / 'base.tar'
        print(f'WARNING: Checkpoint {ckpt} not found, using base.tar instead: {ckpt_path}')
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    if len(unexpected_keys) > 0:
        print('Unexpected keys:', unexpected_keys)
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {ckpt_path}!')
    return t2m_transformer

def load_res_model(res_opt, ckpt, vq_opt, clip_version='ViT-B/32', device='cuda'):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt_path = Path(res_opt.checkpoints_dir) / res_opt.dataset_name / res_opt.name / 'model' / ckpt
    
    if not ckpt_path.exists():
        ckpt_path = Path(res_opt.checkpoints_dir) / res_opt.dataset_name / res_opt.name / 'model' / 'base.tar'
        print(f'WARNING: Checkpoint {ckpt} not found, using base.tar instead: {ckpt_path}')
    
    ckpt = torch.load(ckpt_path,
                      map_location=device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)

    if len(unexpected_keys) > 0:
        print('Unexpected keys:', unexpected_keys)
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {ckpt_path}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(Path(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from {ckpt["epoch"]}!')
    return model


