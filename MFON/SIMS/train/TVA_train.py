import torch
import numpy as np
import datetime
from tqdm import tqdm
from utils import write_log, set_random_seed
from models.model import TVA_fusion
from utils import write_config
from models.classifier import BaseClassifier


def build_css_scores(train_data):
    dataset = getattr(train_data, 'dataset', None)
    if dataset is None or not hasattr(dataset, 'data'):
        return None
    data = dataset.data
    if 'vision' not in data or 'audio' not in data:
        return None
    scores = []
    for i in range(len(dataset)):
        vision = np.asarray(data['vision'][i], dtype=np.float32)
        audio = np.asarray(data['audio'][i], dtype=np.float32)
        vision_score = np.linalg.norm(vision) / np.sqrt(max(1, vision.size))
        audio_score = np.linalg.norm(audio) / np.sqrt(max(1, audio.size))
        scores.append((vision_score + audio_score) / 2.0)
    scores = np.asarray(scores, dtype=np.float32)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return torch.from_numpy(scores)


def get_css_threshold(css_scores, epoch, curriculum_epoch, min_ratio):
    if css_scores is None:
        return None, 1.0
    ratio = min(1.0, max(float(min_ratio), float(epoch) / float(max(1, curriculum_epoch))))
    threshold = float(np.quantile(css_scores.numpy(), 1.0 - ratio))
    return threshold, ratio


def filter_text_batch(text, mask):
    positions = mask.nonzero(as_tuple=False).view(-1).tolist()
    if isinstance(text, (list, tuple)):
        return [text[i] for i in positions]
    if isinstance(text, np.ndarray):
        return text[positions].tolist()
    return text


def TVA_train_fusion(config, metrics, seed, train_data, valid_data):
    print('---------------TVA_EXP---------------')
    
    set_random_seed(seed)
    
    update_epochs = config.SIMS.downStream.update_epochs
    
    text_lr = config.SIMS.downStream.TVAtrain.text_lr
    audio_lr = config.SIMS.downStream.TVAtrain.audio_lr
    vision_lr = config.SIMS.downStream.TVAtrain.vision_lr
    other_lr = config.SIMS.downStream.TVAtrain.other_lr
    
    text_decay = config.SIMS.downStream.TVAtrain.text_decay
    audio_decay = config.SIMS.downStream.TVAtrain.audio_decay
    vision_decay = config.SIMS.downStream.TVAtrain.vision_decay
    other_decay = config.SIMS.downStream.TVAtrain.other_decay
            
    delta_va = config.SIMS.downStream.TVAtrain.delta_va 
    delta_nce = config.SIMS.downStream.TVAtrain.delta_nce
    use_css = getattr(config.SIMS.downStream.TVAtrain, 'use_css', False)
    css_epoch = getattr(config.SIMS.downStream.TVAtrain, 'css_epoch', 20)
    css_min_ratio = getattr(config.SIMS.downStream.TVAtrain, 'css_min_ratio', 0.2)
    
    model = TVA_fusion(config).to(config.DEVICE)
    css_scores = build_css_scores(train_data) if use_css else None
    if css_scores is not None:
        print('CSS enabled: %d samples, min_ratio=%s, curriculum_epoch=%s' % (
            len(css_scores), css_min_ratio, css_epoch
        ))

    model.load_froze()
    
    text_params = list(model.proj_t.named_parameters()) + list(model.text_encoder.named_parameters())
    text_params = [p for _, p in text_params] 
    vision_params = list(model.proj_v.named_parameters()) +\
                list(model.vision_with_text.named_parameters()) 
    vision_params = [p for _, p in vision_params] + [model.promptv_m]
    audio_params = list(model.proj_a.named_parameters()) +\
                list(model.audio_with_text.named_parameters())
    audio_params = [p for _, p in audio_params] + [model.prompta_m]
    model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n] 
    alw_params = [
        model.alpha_base_v,
        model.alpha_base_a,
        model.beta_base,
    ] if getattr(model, 'use_alw', False) else []
    dpg_params = []
    if getattr(model, 'use_dpg', False):
        dpg_params = list(model.dynamic_prompt_v.parameters()) + list(model.dynamic_prompt_a.parameters())

    optimizer_grouped_parameters = [
        {'params': text_params, 'weight_decay': text_decay, 'lr': text_lr},
        {'params': audio_params, 'weight_decay': audio_decay, 'lr': audio_lr},
        {'params': vision_params, 'weight_decay': vision_decay, 'lr': vision_lr},
        {'params': model_params_other + alw_params + dpg_params, 'weight_decay': other_decay, 'lr': other_lr}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
   
    loss, best_loss  = 0, 1e8
    loss_a = loss_v = pred_loss = loss_nce = 0
    device = config.DEVICE  
    total_epoch = config.SIMS.downStream.TVAtrain.epoch
    best_epoch = 1
    for epoch in range(1, total_epoch + 1):
        
        model.train()
        left_epochs = update_epochs
        alw_epoch_stats = {
            'alpha_v': 0.0, 'alpha_a': 0.0, 'beta': 0.0,
            'q_v': 0.0, 'q_a': 0.0, 'progress': 0.0, 'count': 0
        }
        budget_epoch_stats = {
            'q_v': 0.0, 'q_a': 0.0, 'w_v': 0.0, 'w_a': 0.0,
            'q_v_std': 0.0, 'q_a_std': 0.0, 'w_v_std': 0.0, 'w_a_std': 0.0,
            'w_nce_v': 0.0, 'w_nce_a': 0.0, 'loss_v': 0.0,
            'loss_a': 0.0, 'loss_nce': 0.0, 'progress': 0.0, 'count': 0,
        }
        css_threshold, css_ratio = get_css_threshold(css_scores, epoch, css_epoch, css_min_ratio)
        if css_threshold is not None:
            print('CSS epoch %d: keep_ratio=%.4f threshold=%.6f' % (epoch, css_ratio, css_threshold))
        bar = tqdm(train_data, disable=False)
        for index, batch_data in enumerate(bar):
            try:
                bar.set_description("Epoch:%d|loss:%s|pred_loss:%s|loss_v:%s|loss_a:%s|loss_nce:%s" % (
                    epoch, loss.item(), pred_loss.item(), loss_v.item(), loss_a.item(),  loss_nce.item()
                    )
                )
            except:
                bar.set_description(
                    "Epoch:%d|loss:%s|pred_loss:%s|loss_v:%s|loss:%s|loss_nce:%s" % (epoch, loss, pred_loss, loss_v, loss_a, loss_nce)
                )
            if left_epochs == update_epochs:
                optimizer.zero_grad()
            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            label = batch_data['labels']['M'].clone().detach().view(-1).to(device).float()
            if css_scores is not None:
                sample_indices = batch_data['index'].long()
                css_mask = css_scores[sample_indices] >= css_threshold
                if css_mask.sum().item() < 2:
                    continue
                text = filter_text_batch(text, css_mask)
                device_mask = css_mask.to(device)
                vision = vision[device_mask]
                audio = audio[device_mask]
                label = label[device_mask]

            left_epochs -= 1
        
            pred, (loss_v, loss_a, loss_nce) = model(text, vision, audio, mode='train', epoch=epoch)
            
            pred_loss = torch.mean((pred-label)*(pred-label))  # [bs]
            
            if getattr(model, 'use_budgeted_aux', False) and model.current_budgeted_aux is not None:
                budget = model.current_budgeted_aux
                loss = pred_loss + budget['loss']
                for key in ['q_v', 'q_a', 'w_v', 'w_a', 'w_nce_v', 'w_nce_a']:
                    budget_epoch_stats[key] += budget[key].detach().mean().item()
                for key in ['q_v', 'q_a', 'w_v', 'w_a']:
                    budget_epoch_stats[key + '_std'] += budget[key].detach().std(unbiased=False).item()
                for key in ['loss_v', 'loss_a', 'loss_nce', 'progress']:
                    budget_epoch_stats[key] += budget[key].detach().item()
                budget_epoch_stats['count'] += 1
            elif getattr(model, 'use_alw', False) and model.current_alw is not None:
                weights = model.current_alw
                loss = pred_loss + weights['alpha_v'] * loss_v + weights['alpha_a'] * loss_a + weights['beta'] * loss_nce
                for key in ['alpha_v', 'alpha_a', 'beta', 'q_v', 'q_a', 'progress']:
                    alw_epoch_stats[key] += weights[key].detach().item()
                alw_epoch_stats['count'] += 1
            else:
                loss = pred_loss + delta_va * (loss_v + loss_a) + delta_nce * loss_nce
            
            loss.backward()
           
            if not left_epochs:
                optimizer.step()
                left_epochs = update_epochs
                
        if not left_epochs:
            optimizer.step()

        _, result_loss = eval(model, metrics, valid_data, device)
        if alw_epoch_stats['count'] > 0:
            alw_log = {
                key: round(alw_epoch_stats[key] / alw_epoch_stats['count'], 6)
                for key in ['alpha_v', 'alpha_a', 'beta', 'q_v', 'q_a', 'progress']
            }
            print('ALW epoch %d:' % epoch, alw_log)
        if budget_epoch_stats['count'] > 0:
            budget_log = {
                key: round(budget_epoch_stats[key] / budget_epoch_stats['count'], 6)
                for key in [
                    'q_v', 'q_a', 'q_v_std', 'q_a_std',
                    'w_v', 'w_a', 'w_v_std', 'w_a_std', 'w_nce_v', 'w_nce_a',
                    'loss_v', 'loss_a', 'loss_nce', 'progress',
                ]
            }
            print('Budgeted auxiliary epoch %d:' % epoch, budget_log)
        
        if result_loss < best_loss:
            best_loss = result_loss
            model.save_model()


def eval(model, metrics, eval_data, device):
    model.eval()
    with torch.no_grad():
        pred, truth = [], []
        loss = 0
        lens = 0 
        for index, batch_data in enumerate(eval_data):
            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            label = batch_data['labels']['M'].clone().detach().view(-1).to(device).float()
            _pred, _ = model(text, vision, audio, mode='test')
            pred.append(_pred.view(-1))
            truth.append(label)
            _loss = torch.mean((label-_pred)*(label-_pred))
            loss += _loss.item() * len(label)
            lens += len(label)
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu')).squeeze()
        eval_results = metrics.eval_sims_regression(truth, pred)
        eval_results['Loss'] = loss / lens
    model.train()
    return eval_results, loss / lens


def TVA_test_fusion(config, metric, test_data,mode_path=None):
    
    seed = config.seed
    log_path = config.LOGPATH + "SIMS_TVA_Test." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'

    write_config(config, log_path)
    
    model = TVA_fusion(config=config)

    device = config.DEVICE
    model.to(device)
    
    model.load_model(mode_path)
    result, loss = eval(model,metric, test_data, device)
   
    log = '\nTVA_Test result\n\tacc_2:%s\n\tacc_3:%s\n\tacc_7:%s\n\t' \
        'F1_score:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
        '------------------------------------------' % (
        result["Mult_acc_2"], result["Mult_acc_3"],result["Mult_acc_5"], result["F1_score"], 
        result['MAE'], result['Corr'], loss
    )
    write_log(log, log_path)
    print(log)
    
    return result
