import numpy as np
import pandas as pd
import torch, transformers, os
from torch.utils.data import DataLoader
from dataset import RegressionDataset
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml, os, shutil
from model.models import RegressionModelWithResidual
from transformers import RobertaTokenizerFast
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import random


# ==========================================
# 0. 固定随机种子
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 1. 辅助类：标准化工具
# ==========================================
class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, targets):
        self.mean = np.mean(targets, axis=0)
        self.std = np.std(targets, axis=0)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, targets):
        if self.mean is None: raise ValueError("Scaler not fitted")
        return (targets - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, normalized_targets):
        if self.mean is None: return normalized_targets
        return normalized_targets * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# ==========================================
# 2. 训练与验证函数 (静态稳健版)
# ==========================================
def run_epoch(data_loader, model, optimizer, device, criterion, scaler, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = []
    all_preds = []
    all_targets = []

    pbar = tqdm(data_loader, desc="Train" if is_train else "Val", leave=False, disable=not is_train)

    # 【回归最稳配置】：手动静态权重对齐
    # 既然 T2 的量级和 MAE 约为 T1 的两倍，我们给 T2 减半权重，强制量级对齐
    weight_t1 = 1.0
    weight_t2 = 0.5

    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v

            targets = batch_on_device["target"]

            try:
                pred_targets, _ = model(batch_on_device)
            except ValueError as e:
                if "Deep Fusion Model requires 'graph_embs'" in str(e):
                    print(f"FATAL ERROR: {e}")
                    raise

            # 计算 HuberLoss
            loss_raw = criterion(pred_targets, targets)

            # 使用静态权重，彻底杜绝 Batch 间的梯度剧烈跳动
            loss_t1 = loss_raw[:, 0].mean() * weight_t1
            loss_t2 = loss_raw[:, 1].mean() * weight_t2

            loss = loss_t1 + loss_t2

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：限制为 1.0，防止“震飞”
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss.append(loss.item())
            all_preds.append(pred_targets.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    real_preds = scaler.inverse_transform(all_preds)
    real_targets = scaler.inverse_transform(all_targets)

    metrics = {'loss': np.mean(epoch_loss)}
    metrics['mae_t1'] = mean_absolute_error(real_targets[:, 0], real_preds[:, 0])
    metrics['r2_t1'] = r2_score(real_targets[:, 0], real_preds[:, 0])
    metrics['mae_t2'] = mean_absolute_error(real_targets[:, 1], real_preds[:, 1])
    metrics['r2_t2'] = r2_score(real_targets[:, 1], real_preds[:, 1])

    return metrics, real_targets, real_preds


# ==========================================
# 3. 单次训练流程
# ==========================================
def train_one_fold(seed, fold, config, df_train, df_val, scaler, run_dir):
    print(f"\n🌱 Starting Seed {seed} | Fold {fold}...")
    seed_everything(seed + fold)

    DEVICE = config["device"]
    if config.get("debug", False): DEVICE = "cpu"

    EQ_COL_NAME = 'eq_emb'
    train_graph_embs = None
    val_graph_embs = None
    current_graph_dim = 128

    for df, label in [(df_train, 'Train'), (df_val, 'Val')]:
        has_eq = False
        if EQ_COL_NAME in df.columns and not df[EQ_COL_NAME].isnull().all():
            try:
                valid_embs = np.stack(df[EQ_COL_NAME].dropna().values)
                if len(valid_embs) == len(df):
                    if label == 'Train':
                        train_graph_embs = valid_embs
                        current_graph_dim = train_graph_embs.shape[1]
                    else:
                        val_graph_embs = valid_embs
                    has_eq = True
            except Exception as e:
                print(f"⚠️ Error loading eq_emb in {label}: {e}")
        if not has_eq:
            zeros = np.zeros((len(df), current_graph_dim), dtype=np.float32)
            if label == 'Train':
                train_graph_embs = zeros
            else:
                val_graph_embs = zeros

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    m_len = int(tokenizer.model_max_length)

    train_ds = RegressionDataset(texts=df_train["text"].values,
                                 targets=np.stack(df_train["target_norm"].values),
                                 tokenizer=tokenizer,
                                 graph_embs=train_graph_embs,
                                 seq_len=m_len)

    val_ds = RegressionDataset(texts=df_val["text"].values,
                               targets=np.stack(df_val["target_norm"].values),
                               tokenizer=tokenizer,
                               graph_embs=val_graph_embs,
                               seq_len=m_len)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    with open(config["model_config"], "r") as f:
        m_config = yaml.safe_load(f)

    model = RegressionModelWithResidual(m_config, graph_dim=current_graph_dim).to(DEVICE)

    if config.get("pt_ckpt_path"):
        ckpt = torch.load(config['pt_ckpt_path'], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)

    # 【稳健核心】：delta=1.0 调回标准值，保证低误差区的训练强度
    loss_fn = nn.HuberLoss(reduction='none', delta=1.0).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if "text_encoder" in n], 'lr': 3e-5},
        {'params': [p for n, p in model.named_parameters() if "text_encoder" not in n], 'lr': 8e-4}
    ], weight_decay=0.01)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_loss = 999
    early_stop = 0

    for epoch in range(1, config['num_epochs'] + 1):
        tr_m, _, _ = run_epoch(train_loader, model, optimizer, DEVICE, loss_fn, scaler, True)
        val_m, _, _ = run_epoch(val_loader, model, optimizer, DEVICE, loss_fn, scaler, False)

        scheduler.step(val_m['loss'])

        if val_m['loss'] < best_loss:
            best_loss = val_m['loss']
            early_stop = 0
            torch.save({'model_state_dict': model.state_dict(), 'scaler_state_dict': scaler.state_dict()},
                       os.path.join(run_dir, f'best_model_seed{seed}_fold{fold}.pt'))
        else:
            early_stop += 1
        if early_stop >= config['early_stop_threshold']: break

    return best_loss


# ==========================================
# 4. 主入口
# ==========================================
def run_ensemble(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config.update({'num_epochs': 120, 'batch_size': 16, 'early_stop_threshold': 20, 'device': 'cuda'})

    RUN_NAME = f"{config['run_name']}_BACK_TO_BASICS_STABLE_{datetime.now().strftime('%m%d_%H%M')}"
    CKPT_SAVE_DIR = os.path.join(config["ckpt_save_path"], RUN_NAME)
    if not os.path.exists(CKPT_SAVE_DIR): os.makedirs(CKPT_SAVE_DIR)

    df_train_raw = pd.read_pickle(config["train_path"])
    df_val_raw = pd.read_pickle(config["val_path"])
    df_all = pd.concat([df_train_raw, df_val_raw], axis=0).reset_index(drop=True)

    SEEDS = [42, 43, 44, 45, 46]
    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df_all)):
            df_train = df_all.iloc[train_idx].copy()
            df_val = df_all.iloc[val_idx].copy()

            train_targets = df_train[['target_1', 'target_2']].values.astype(float)
            scaler = TargetScaler()
            scaler.fit(train_targets)

            df_train["target_norm"] = list(scaler.transform(train_targets))
            df_val["target_norm"] = list(scaler.transform(df_val[['target_1', 'target_2']].values.astype(float)))

            train_one_fold(seed, fold, config, df_train, df_val, scaler, CKPT_SAVE_DIR)


if __name__ == "__main__":
    run_ensemble("regress_train.yml")