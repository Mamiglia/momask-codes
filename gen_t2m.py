import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from .models.loaders import load_res_model, load_trans_model, load_vq_model, load_len_estimator
from .models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from .models.vq.model import RVQVAE, LengthEstimator

from .options.eval_option import EvalT2MOptions
from .utils.get_opt import get_opt

from .utils.fixseed import fixseed
from .visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical

import wandb

from .utils.motion_process import recover_from_ric
from .utils.plot_script import plot_3d_motion

from .utils.paramUtil import t2m_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

def add_viz_args(parser):
    parser.add_argument('--ckpt', type=str, default='latest.tar', help='Checkpoint file to load')
    parser.add_argument('--skip_viz', action='store_true', help='Skip visualization')
    parser.add_argument('--ik_viz', action='store_true', help='Use IK for visualization')
    parser.add_argument('--run_name', type=str, default='t2m_gen', help='Name of the run for wandb logging')


if __name__ == '__main__':
    parser = EvalT2MOptions()
    add_viz_args(parser.parser)
    opt = parser.parse()
    fixseed(opt.seed)
    
    wandb.init(
        resume='allow'
    )

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.run_name)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    feats_dir = pjoin(result_dir, 'feats')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)
    os.makedirs(feats_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)


    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt, ckpt=opt.ckpt, device=opt.device)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, opt.ckpt, vq_opt, device=opt.device)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt.ckpt, device=opt.device)

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    est_length = False
    if opt.text_prompt != "":
        prompt_list.append(opt.text_prompt)
        if opt.motion_length == 0:
            est_length = True
        else:
            length_list.append(opt.motion_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[1]))
    else:
        raise "A text prompt, or a file a text prompts are required!!!"
    # print('loading checkpoint {}'.format(file))

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = t2m_transformer.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
        # lengths = torch.multinomial()
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list

    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids = t2m_transformer.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)
            # print(mids)
            # print(mids.shape)
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            pred_motions = vq_model.forward_decoder(mids)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)
            
        for k, (caption, feats)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            name = '_'.join(caption.strip().split(' ')[-3:]) + f'_len{m_length[k]}'

            feats = feats[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(feats).float(), 22).numpy()
            np.save(pjoin(joints_dir, f"{name}.npy"), joint)
            np.save(pjoin(feats_dir, f"{name}.npy"), feats)

            if opt.skip_viz:
                continue
            bvh_dir = pjoin(animation_dir, f"{name}.bvh")
            _, joint = converter.convert(joint, filename=bvh_dir, iterations=100, foot_ik=False)

            save_dir = pjoin(animation_dir, f"{name}.mp4")
            plot_3d_motion(save_dir, kinematic_chain, joint, title=caption, fps=20)

            wandb.log({
                f'video/{name}': wandb.Video(save_dir, caption=caption), 
                f'video/{name}/caption':caption,
                f'video/{name}/name':name,
                f'video/{name}/length': m_length[k],
                })

            if opt.ik_viz:
                bvh_dir = pjoin(animation_dir, f"{name}_ik.bvh")
                _, ik_joint = converter.convert(joint, filename=bvh_dir, iterations=100)
                ik_save_dir = pjoin(animation_dir, f"{name}_ik.mp4")
                plot_3d_motion(ik_save_dir, kinematic_chain, ik_joint, title=caption, fps=20)
                np.save(pjoin(joints_dir, f"{name}_ik.npy"), ik_joint)