import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

from budgeted_auxiliary import (
    build_budgeted_auxiliary,
    per_sample_infonce,
    per_sample_kl,
)

import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
sys.path.append(os.path.dirname(path))

from classifier import BaseClassifier
from Text_encoder import TextEncoder
from Vision_encoder import VisionEncoder
from Audio_encoder  import AudioEncoder
from models.trans.transformer import TransformerEncoder
from models.classifier import BaseClassifier


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def inverse_softplus(value):
    value = max(float(value), 1e-6)
    return math.log(math.exp(value) - 1.0)


class DynamicPromptGate(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, context_dim),
            nn.Sigmoid(),
        )

    def forward(self, text_embed, modal_embed, static_prompt):
        text_context = text_embed.mean(dim=0)
        modal_context = modal_embed.mean(dim=0)
        gate = self.net(torch.cat([text_context, modal_context], dim=-1))
        return static_prompt.unsqueeze(1) * gate.unsqueeze(0)

class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))



class TVA_fusion(nn.Module):
    def __init__(self, config):
        super(TVA_fusion, self).__init__()
        self.config = config
        
        self.text_dropout = config.MOSEI.downStream.text_drop_out
        
        encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        audio_text_nhead = config.MOSEI.downStream.audio_text_nhead
        audio_text_tf_num_layers = config.MOSEI.downStream.audio_text_tf_num_layers
        
        self.audio_fea_dim = config.MOSEI.downStream.audio_fea_dim
        self.a_len = config.MOSEI.downStream.audio_seq_len
        
        vision_text_nhead = config.MOSEI.downStream.vision_text_nhead
        vision_text_tf_num_layers = config.MOSEI.downStream.vision_text_tf_num_layers
        
        attn_dropout= config.MOSEI.downStream.drop_out
        attn_mask = config.MOSEI.downStream.attn_mask
        
        audio_fea_dim = config.MOSEI.downStream.audio_fea_dim
        vision_fea_dim = config.MOSEI.downStream.vision_fea_dim
        text_fea_dim = config.MOSEI.downStream.text_fea_dim
        
        self.vlen, self.alen = config.MOSEI.downStream.vlen, config.MOSEI.downStream.alen
        
        self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))
        self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim)) 
        
        self.text_encoder = TextEncoder(config=config)
        self.proj_t = nn.Linear(text_fea_dim, encoder_fea_dim)
        
        self.proj_v = nn.Linear(vision_fea_dim, encoder_fea_dim)
        
        self.vision_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=vision_text_nhead, layers=vision_text_tf_num_layers, 
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        ) # Q:text, KV:vision
        
        self.proj_a = nn.Linear(audio_fea_dim, encoder_fea_dim)
        self.audio_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=audio_text_nhead, layers=audio_text_tf_num_layers, 
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        ) # Q:text, KV:audio
        
        self.vision_encoder_froze = VisionEncoder(config=config)
        self.audio_encoder_froze = AudioEncoder(config=config)
        
        self.TVA_decoder = BaseClassifier(
            input_size=encoder_fea_dim * 3,
            hidden_size=[encoder_fea_dim, encoder_fea_dim//2, encoder_fea_dim//8],
            output_size=1
        )
        self.device = config.DEVICE
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)
        train_cfg = config.MOSEI.downStream.TVAtrain
        exp_name = getattr(train_cfg, 'exp_name', None)
        if exp_name:
            self.model_path = os.path.join(self.model_path, exp_name) + '/'
            check_dir(self.model_path)
        self.use_alw = getattr(train_cfg, 'use_alw', False)
        self.alw_warmup_epoch = max(1, getattr(train_cfg, 'alw_warmup_epoch', 10))
        self.alw_q_type = getattr(train_cfg, 'alw_q_type', 'align')
        self.alw_temperature = getattr(train_cfg, 'alw_temperature', 1.0)
        self.use_budgeted_aux = getattr(train_cfg, 'use_budgeted_aux', False)
        self.budget_warmup_epoch = max(1, getattr(train_cfg, 'budget_warmup_epoch', 10))
        self.budget_epsilon = getattr(train_cfg, 'budget_epsilon', 1e-8)
        if self.use_alw and self.use_budgeted_aux:
            raise ValueError('use_alw and use_budgeted_aux are mutually exclusive.')
        self.alpha_base_v = nn.Parameter(torch.tensor(inverse_softplus(train_cfg.delta_va)))
        self.alpha_base_a = nn.Parameter(torch.tensor(inverse_softplus(train_cfg.delta_va)))
        self.beta_base = nn.Parameter(torch.tensor(inverse_softplus(train_cfg.delta_nce)))
        self.current_alw = None
        self.current_budgeted_aux = None
        self.use_dpg = getattr(train_cfg, 'use_dpg', False)
        dpg_hidden_dim = getattr(train_cfg, 'dpg_hidden_dim', 256)
        self.dynamic_prompt_v = DynamicPromptGate(encoder_fea_dim, dpg_hidden_dim)
        self.dynamic_prompt_a = DynamicPromptGate(encoder_fea_dim, dpg_hidden_dim)
     
    def load_froze(self):
        model_path = self.config.MOSEI.path.encoder_path + str(self.config.seed) + '/'
        self.audio_encoder_froze.load_state_dict(torch.load(model_path+ 'best_loss_audio_encoder.pt', map_location=self.device))
        self.vision_encoder_froze.load_state_dict(torch.load(model_path + 'best_loss_vision_encoder.pt', map_location=self.device))
        self.audio_encoder_froze.set_froze()
        self.vision_encoder_froze.set_froze()
       
    def forward(self, text, vision, audio, mode='train', epoch=None):
        self.current_alw = None
        self.current_budgeted_aux = None
        last_hidden_text  = self.text_encoder(text)   # [bs, seq, h] [bs, h]
        last_hidden_text = F.dropout(self.proj_t(last_hidden_text.permute(1, 0, 2)), 
            p=self.text_dropout, training=self.training
        )
        x_t_embed = last_hidden_text[0]
        
        proj_vision_base = self.proj_v(vision).permute(1, 0, 2)
        if self.use_dpg:
            prompt_v = self.dynamic_prompt_v(last_hidden_text, proj_vision_base, self.promptv_m)
        else:
            prompt_v = self.promptv_m.unsqueeze(1)
        proj_vision = proj_vision_base + prompt_v
        h_tv = self.vision_with_text(last_hidden_text, proj_vision, proj_vision)    # [seq-v, bs, 768] [seq-t, bs, 768]--> [seq, bs,h]
        
        proj_audio_base = self.proj_a(audio).permute(1, 0, 2)
        if self.use_dpg:
            prompt_a = self.dynamic_prompt_a(last_hidden_text, proj_audio_base, self.prompta_m)
        else:
            prompt_a = self.prompta_m.unsqueeze(1)
        proj_audio = proj_audio_base + prompt_a
        h_ta = self.audio_with_text(last_hidden_text, proj_audio, proj_audio)
        
        x_v_embed = h_tv[0]   
        
        x_a_embed = h_ta[0]   
        
        x = torch.cat([x_t_embed, x_v_embed, x_a_embed], dim=-1) # [bs, 3h]
        pred = self.TVA_decoder(x).view(-1) # [bs]
        
        loss_v = loss_a = 0
        loss_nce = 0
        if mode == 'train':
            x_v_embed_froze = self.vision_encoder_froze(vision)
            x_a_embed_froze = self.audio_encoder_froze(audio)  # [bs, h], [bs, h]
            if self.use_budgeted_aux:
                loss_v_each = per_sample_kl(x_v_embed, x_v_embed_froze)
                loss_a_each = per_sample_kl(x_a_embed, x_a_embed_froze)
                loss_nce_v_each = per_sample_infonce(x_v_embed, x_t_embed)
                loss_nce_a_each = per_sample_infonce(x_a_embed, x_t_embed)
                loss_v = loss_v_each.mean()
                loss_a = loss_a_each.mean()
                loss_nce = loss_nce_v_each.mean() + loss_nce_a_each.mean()
                self.current_budgeted_aux = self.get_budgeted_auxiliary(
                    epoch,
                    x_v_embed,
                    x_a_embed,
                    x_v_embed_froze,
                    x_a_embed_froze,
                    pred,
                    loss_v_each,
                    loss_a_each,
                    loss_nce_v_each,
                    loss_nce_a_each,
                )
            else:
                loss_v = self.get_KL_loss(x_v_embed, x_v_embed_froze)
                loss_a = self.get_KL_loss(x_a_embed, x_a_embed_froze)
                loss_nce = self.get_InfoNCE_loss(x_v_embed, x_t_embed) + self.get_InfoNCE_loss(x_a_embed, x_t_embed)
            self.current_alw = self.get_alw_weights(
                epoch, x_v_embed, x_a_embed, x_v_embed_froze, x_a_embed_froze, pred
            )
        else:
            return pred, None
            
        return pred, (loss_v, loss_a, loss_nce)
        
    def save_model(self,name=None):
        # save all modules
        if name==None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = self.model_path + str(name)+'TVA_fusion' + '_model.pt'
        print('model saved at:\n', mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name=None):
        if name==None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = name
        print('model loaded from:\n', mode_path)
        # self.load_state_dict(torch.load(mode_path, map_location=self.device))
        checkpoint = torch.load(mode_path, map_location=self.device)
        model_state_dict = self.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        self.load_state_dict(filtered_checkpoint, strict=False)

    def get_distill_loss(self, input1, input2):
        diff_loss = torch.mean((input1-input2)*(input1-input2))
        return diff_loss
    
    def get_KL_loss(self, x_embed, x_embed_target):
        x_embed1 = F.log_softmax(x_embed, dim=1)
        x_embed_target1 = F.softmax(x_embed_target, dim=1)
        loss = self.criterion(x_embed1, x_embed_target1)
        return loss
    
    def get_InfoNCE_loss(self, input1, input2):
        
        x1 = input1 / input1.norm(dim=1, keepdim=True)
        x2 = input2 / input2.norm(dim=1, keepdim=True)

        pos = torch.sum(x1*x2, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x1, x2.t()), dim=-1)   # bs
        nce_loss = -(pos - neg).mean()
        
        return nce_loss

    def get_quality_score(self, embed, target_embed, pred=None):
        if self.alw_q_type == 'norm':
            dim_scale = math.sqrt(max(1, embed.size(-1)))
            quality = torch.sigmoid(embed.detach().norm(dim=1).mean() / (dim_scale * self.alw_temperature))
        elif self.alw_q_type == 'conf':
            if pred is None:
                quality = torch.tensor(1.0, device=embed.device)
            else:
                quality = torch.sigmoid(pred.detach().abs().mean() / self.alw_temperature)
        else:
            quality = (F.cosine_similarity(embed.detach(), target_embed.detach(), dim=1).mean() + 1.0) / 2.0
        return quality.clamp(0.0, 1.0)

    def get_alw_weights(self, epoch, x_v_embed, x_a_embed, x_v_target, x_a_target, pred=None):
        if not self.use_alw:
            return None
        epoch = 1 if epoch is None else epoch
        progress = min(float(epoch) / float(self.alw_warmup_epoch), 1.0)
        progress = torch.tensor(progress, device=x_v_embed.device)
        q_v = self.get_quality_score(x_v_embed, x_v_target, pred)
        q_a = self.get_quality_score(x_a_embed, x_a_target, pred)
        alpha_v = F.softplus(self.alpha_base_v) * progress * q_v
        alpha_a = F.softplus(self.alpha_base_a) * progress * q_a
        beta = F.softplus(self.beta_base) * progress * (q_v + q_a) / 2.0
        return {
            'alpha_v': alpha_v,
            'alpha_a': alpha_a,
            'beta': beta,
            'q_v': q_v,
            'q_a': q_a,
            'progress': progress,
        }

    def get_budgeted_auxiliary(
        self,
        epoch,
        x_v_embed,
        x_a_embed,
        x_v_target,
        x_a_target,
        pred,
        loss_v_each,
        loss_a_each,
        loss_nce_v_each,
        loss_nce_a_each,
    ):
        epoch = 1 if epoch is None else epoch
        progress = min(float(epoch) / float(self.budget_warmup_epoch), 1.0)
        train_cfg = self.config.MOSEI.downStream.TVAtrain
        return build_budgeted_auxiliary(
            x_v_embed,
            x_a_embed,
            x_v_target,
            x_a_target,
            pred,
            loss_v_each,
            loss_a_each,
            loss_nce_v_each,
            loss_nce_a_each,
            self.alw_q_type,
            self.alw_temperature,
            train_cfg.delta_va,
            train_cfg.delta_nce,
            progress,
            self.budget_epsilon,
        )
