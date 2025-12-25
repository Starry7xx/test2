import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.models import RegressionModel, RegressionModel2, RegressionModelWithResidual
import numpy as np
import pandas as pd
import os, yaml, pickle, glob
from transformers import RobertaTokenizerFast
import tqdm
import argparse
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error


# -------------------------------

class InferenceScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def inverse_transform(self, normalized_preds):
        if self.mean is None: return normalized_preds
        return normalized_preds * self.std + self.mean


def predict_fn(data_loader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="    Inferring", leave=False):
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v

            outputs_raw = model(batch_on_device)
            outputs = outputs_raw[0] if isinstance(outputs_raw, tuple) else outputs_raw
            if outputs.shape[-1] == 1: outputs = outputs.squeeze(-1)
            predictions.extend(outputs.cpu().numpy())
    return np.array(predictions)


def run_ensemble_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug=False):
    print("=============================================================")
    print(f"Ensemble Prediction scanning: {pt_ckpt_dir_path}")
    print("=============================================================")

    # 1. 扫描所有模型权重
    seed_ckpts = glob.glob(os.path.join(pt_ckpt_dir_path, "best_model_seed*_fold*.pt"))

    if len(seed_ckpts) == 0:
        seed_ckpts = glob.glob(os.path.join(pt_ckpt_dir_path, "best_model_seed_*.pt"))

    if len(seed_ckpts) == 0:
        fallback = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")
        if os.path.exists(fallback):
            seed_ckpts = [fallback]
        else:
            raise FileNotFoundError(f"No checkpoints found in {pt_ckpt_dir_path}!")

    print(f"🔎 Found {len(seed_ckpts)} models to ensemble.")

    # A. 加载 Config
    model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml")
    if not os.path.exists(model_config_path):
        model_config_path = os.path.join(os.path.dirname(pt_ckpt_dir_path), "clip.yml")
        if not os.path.exists(model_config_path):
            model_config_path = "../model/clip.yml"

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # B. 预检查模型架构
    try:
        sample_ckpt = torch.load(seed_ckpts[0], map_location='cpu', weights_only=False)
    except:
        sample_ckpt = torch.load(seed_ckpts[0], map_location='cpu')

    sample_sd = sample_ckpt['model_state_dict'] if 'model_state_dict' in sample_ckpt else sample_ckpt

    required_graph_dim = 0
    if any("regresshead.weight" in k for k in sample_sd.keys()):
        w_key = next(k for k in sample_sd.keys() if k.endswith('regresshead.weight'))
        input_dim = sample_sd[w_key].shape[1]
        roberta_dim = model_config['RobertaConfig']['hidden_size']
        if input_dim > roberta_dim:
            required_graph_dim = input_dim - roberta_dim
            print(f"    Detected Deep Fusion Model (Expects {required_graph_dim} dim Graph Features)")

    # 2. 准备数据
    df_test = pd.read_pickle(data_path)
    if debug: df_test = df_test.sample(10)

    device = "cuda" if torch.cuda.is_available() and not debug else "cpu"
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    EQ_COL_NAME = 'eq_emb'
    test_graph_embs = None
    has_real_data = False
    if EQ_COL_NAME in df_test.columns and not df_test[EQ_COL_NAME].isnull().all():
        valid_embs = df_test[EQ_COL_NAME].dropna().values
        if len(valid_embs) == len(df_test):
            try:
                test_graph_embs = np.stack(valid_embs)
                has_real_data = True
            except:
                pass

    if not has_real_data and required_graph_dim > 0:
        test_graph_embs = np.zeros((len(df_test), required_graph_dim), dtype=np.float32)

    # 【重要修改】：适配多列 target 名字
    target_cols = ['target_1', 'target_2']
    has_targets = all(c in df_test.columns for c in target_cols)

    targets_real = None
    if has_targets:
        print(f"✅ Found targets {target_cols}. Statistical report will be generated.")
        targets_real = df_test[target_cols].values.astype(float)
    else:
        print(f"⚠️ Targets {target_cols} not fully found. Skipping statistical report.")

    test_ds = RegressionDataset(texts=df_test["text"].values,
                                targets=targets_real if has_targets else np.zeros((len(df_test), 2)),
                                tokenizer=tokenizer, seq_len=tokenizer.model_max_length,
                                graph_embs=test_graph_embs)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # 4. 循环预测并累加
    accumulated_preds = None
    scaler = None

    # 用于收集每个模型独立指标的容器
    indiv_mae_list = []
    indiv_r2_list = []

    for i, ckpt_path in enumerate(seed_ckpts):
        print(f"🤖 Model {i + 1}/{len(seed_ckpts)}: {os.path.basename(ckpt_path)}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except:
            checkpoint = torch.load(ckpt_path, map_location=device)

        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        if scaler is None:
            if 'scaler_state_dict' in checkpoint:
                scaler = InferenceScaler(mean=checkpoint['scaler_state_dict']['mean'],
                                         std=checkpoint['scaler_state_dict']['std'])

        # 初始化模型
        w_key = next(k for k in state_dict.keys() if k.endswith('regresshead.weight'))
        current_in_dim = state_dict[w_key].shape[1]
        if current_in_dim > model_config['RobertaConfig']['hidden_size']:
            model = RegressionModelWithResidual(model_config, graph_dim=current_in_dim - roberta_dim).to(device)
        else:
            model = RegressionModel(model_config).to(device)

        model.load_state_dict(state_dict, strict=False)
        preds_norm = predict_fn(test_loader, model, device)

        # 统计逻辑：计算当前独立模型在真实标签下的性能
        if has_targets:
            p_real = scaler.inverse_transform(preds_norm) if scaler else preds_norm
            indiv_mae_list.append(
                [mean_absolute_error(targets_real[:, j], p_real[:, j]) for j in range(targets_real.shape[1])])
            indiv_r2_list.append([r2_score(targets_real[:, j], p_real[:, j]) for j in range(targets_real.shape[1])])

        if accumulated_preds is None:
            accumulated_preds = preds_norm
        else:
            accumulated_preds += preds_norm

    # 5. 后处理和保存
    avg_preds = accumulated_preds / len(seed_ckpts)
    final_preds = scaler.inverse_transform(avg_preds) if scaler else avg_preds

    if not os.path.exists(save_path): os.makedirs(save_path)
    save_file = os.path.join(save_path, f"ENSEMBLE-{tag}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(dict(zip(df_test["id"].values, final_preds)), f)
    print(f"💾 Predictions saved to: {save_file}")

    # 输出 均值 ± 标准差 统计报告
    if has_targets and len(indiv_mae_list) > 0:
        print("\n" + "=" * 50)
        print(f"📊 Statistical Report (Mean ± SD) over {len(seed_ckpts)} Models:")
        print("=" * 50)
        indiv_mae_list = np.array(indiv_mae_list)
        indiv_r2_list = np.array(indiv_r2_list)
        for j in range(targets_real.shape[1]):
            col_name = target_cols[j]
            print(f"Target: {col_name}")
            print(f"    MAE: {np.mean(indiv_mae_list[:, j]):.4f} ± {np.std(indiv_mae_list[:, j]):.4f}")
            print(f"    R2 : {np.mean(indiv_r2_list[:, j]):.4f} ± {np.std(indiv_r2_list[:, j]):.4f}")
        print("=" * 50)

    # Ensemble 结果对比
    if has_targets:
        print("\n🏆 Final Ensemble Performance (Averaged Result):")
        for i in range(targets_real.shape[1]):
            r2 = r2_score(targets_real[:, i], final_preds[:, i])
            mae = mean_absolute_error(targets_real[:, i], final_preds[:, i])
            print(f"    {target_cols[i]}: R2 = {r2:.4f} | MAE = {mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pt_ckpt_dir_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.tag is None: args.tag = datetime.now().strftime("%m%d_%H%M")
    run_ensemble_prediction(args.data_path, args.pt_ckpt_dir_path, args.save_path, args.tag, args.debug)