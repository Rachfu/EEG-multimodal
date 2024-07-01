import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import numpy as np
from data import get_data
from model import get_model
from collections import defaultdict

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


set_seed(980616)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=str, default='test')
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--batch_size", "-bs", type=int, default=8)
    parser.add_argument("--data_name", "-d", type=str, default="EEG")
    parser.add_argument("--eps", "-e", type=float, default=2.0)

    parser.add_argument("--n_class", "-c", type=int, default=2)
    parser.add_argument("--n_dp", "-nd", type=int, default=1)
    parser.add_argument("--n_para", "-np", type=int, default=1)
    parser.add_argument("--n_eval", "-ne", type=int, default=5)
    parser.add_argument("--n_epochs", "-n", type=int, default=50)
    parser.add_argument("--interval", type=int, default=1)
    # split by ',' and value should in torchmetric docs, e.g. F1Score,Accuracy,AUROC
    # see https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html for details
    parser.add_argument("--metrics", "-m", type=str, default='Accuracy') 
     
    cfg = parser.parse_args()
    base_path = f'experiment/{cfg.exp}/{cfg.name}/'

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(base_path + 'debug.log', 'w'),
                  logging.FileHandler(base_path + 'info.log', 'w'),
                  logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger()
    logger.handlers[1].setLevel(logging.INFO)
    # logger.handlers[2].setLevel(logging.INFO)
    logger.info(cfg)

    train_loader, val_loader = get_data(cfg)
    model = get_model(cfg)

    criterion = nn.CrossEntropyLoss(reduction='none')

    DP_params = [p for n, p in model.named_parameters() if 'DP' in n]
    model_params = [p for n, p in model.named_parameters() if 'DP' not in n]

    DP_optimizer = optim.SGD(DP_params, lr=0.01)
    model_optimizer = optim.SGD(model_params, lr=0.001)
    results = defaultdict(list)
    metrics = {i: torchmetrics.__dict__[i](task="multiclass", num_classes=cfg.n_class).cuda()
               for i in cfg.metrics.split(',')}
    best_acc = 0.0

    for inputs, labels in val_loader:
        results['labels'].append(labels.cuda().view(-1))
        results['inputs'].append(inputs)
    results['labels'] = torch.cat(results['labels'])

    for epoch in range(cfg.n_epochs):
        model.train()
        train_logits = []
        val_logits = []

        # Training phase
        for i, (inputs, labels) in enumerate(train_loader):
            if i >= 5: break
            if isinstance(inputs, list): inputs = list(i.cuda() for i in inputs)
            else: inputs = inputs.cuda()
            labels = labels.view(-1).cuda()

            # train DP params
            DP_optimizer.zero_grad()
            for _ in range(cfg.n_dp):
                loss = criterion(model(inputs, hard=False), labels)
                loss.sum().backward()
            DP_optimizer.step()

            # train model params
            model_optimizer.zero_grad()
            for _ in range(cfg.n_para):
                loss = criterion(logits := model(inputs, hard=True), labels)
                loss.sum().backward()
                results['train_loss'].append(loss)
            model_optimizer.step()
            logger.debug(f'Train Epoch: {epoch:3d} [{i + 1:3d}/{len(train_loader):3d}]' \
                         f" loss {torch.cat(results['train_loss'][-cfg.n_para:]).mean().item():.4f}")
            
        # Evaluation phase
        if (epoch + 1) % cfg.interval == 0:
            model.eval()
            eval_metrics = defaultdict(list)
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_loader):
                    if isinstance(inputs, list): inputs = list(i.cuda() for i in inputs)
                    else: inputs = inputs.cuda()
                    labels = labels.view(-1).cuda()
                    for _ in range(cfg.n_eval):
                        logits = model(inputs, hard=True)
                        pred = logits.max(1)[1]
                        eval_metrics['logits'].append(logits)
                        eval_metrics['pred'].append(pred)
                        eval_metrics['val_loss'].append(criterion(logits, labels))
                info = f'Eval  Epoch: {epoch:3d}'
                for k, v in eval_metrics.items():
                    results[k].append(torch.cat(v).view(-1, cfg.n_eval, *v[-1].shape[1:]))
                for m, func in metrics.items():
                    vl = [func(results['pred'][-1][:, i], results['labels']) for i in range(cfg.n_eval)]
                    results[m].append(v := torch.tensor(vl))
                    info += f' | {m}: {v.mean().item():5.2f}'
                results['DP_params'].append(model.DP.data)
                logger.info(info)
                if (acc := results['Accuracy'][-1].mean()) > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join(base_path, 'model.pth'))
        torch.save({k: torch.stack(v) for k, v in results.items() if k not in ('labels', 'inputs')}, 
                   os.path.join(base_path, 'results.pth'))
        
        res = torch.load(os.path.join(base_path, 'results.pth'))
        for k, v in res.items():
            print(k, end=' : ')
            if isinstance(v, list): print(len(v))
            else: print(v.shape)
