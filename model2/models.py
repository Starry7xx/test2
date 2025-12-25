import torch
from torch import nn
import torch.nn.functional as F

from .modules import TextEncoder, ProjectionHead, RegressionHead


# [注意] EQCorrectionHead 类已移除，因为不再使用浅层修正


class CLIPModel(nn.Module):
    # ... (CLIPModel, cross_entropy, RegressionModel 保持不变) ...
    def __init__(
            self,
            config
            # temperature=CFG.temperature,
            # image_embedding=CFG.image_embedding,
            # text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        # self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(config)
        # self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(config)
        self.temperature = config['CLIPConfig']['temperature']

    def forward(self, batch):
        # Getting Graph and Text Features
        # image_features = self.image_encoder(batch["image"])
        # breakpoint()
        text_features = self.text_encoder(batch)
        # Getting Image and Text Embeddings (with same dimension)
        # image_embeddings = self.image_projection(image_features)

        text_embeddings = self.text_projection(text_features)
        graph_embeddings = batch['graph_embed']

        # Calculating the Loss
        logits = (text_embeddings @ graph_embeddings.T) / self.temperature
        graphs_similarity = graph_embeddings @ graph_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class RegressionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_encoder = TextEncoder(config)
        self.text_projection = ProjectionHead(config)
        self.regressor = RegressionHead(config)
        self._initialize_weights(self.regressor)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, batch):
        output = self.text_encoder(batch)
        output = self.text_projection(output)
        output = self.regressor(output)
        return output


class RegressionModel2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_encoder = TextEncoder(config)
        # self.text_projection = ProjectionHead(config)
        # self.regressor = RegressionHead(config)
        self.dense = nn.Linear(config['RobertaConfig']['hidden_size'],
                               config['RobertaConfig']['hidden_size'])
        self.activation = nn.Tanh()

        # =========== 输出维度 2 ===========
        self.regresshead = nn.Linear(config['RobertaConfig']['hidden_size'], 2)
        # =============================================

        # self._initialize_weights(self.regressor)
        self._initialize_weights(self.regresshead)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, batch):
        output = self.text_encoder(batch)
        # ========= pooler layer ===========
        output = self.dense(output)
        output = self.activation(output)
        # ==================================
        output = self.regresshead(output)
        return output


# ==========================================
# [最终版本] RegressionModelWithResidual: 深度融合 (Concatenation Fusion)
# ==========================================
class RegressionModelWithResidual(RegressionModel2):

    def __init__(self, config, graph_dim):
        # 初始化 TextEncoder, dense, activation
        super().__init__(config)

        # [修改 1] 重新定义回归头 (regresshead) 的输入维度
        # 新维度 = 文本隐藏层维度 (768) + EQ 嵌入维度 (graph_dim)
        new_input_dim = config['RobertaConfig']['hidden_size'] + graph_dim

        # 重新定义 regresshead 以适应新的输入维度
        self.regresshead = nn.Linear(new_input_dim, 2)
        self._initialize_weights(self.regresshead)  # 重新初始化权重

        print(f"Regression Head Input Dim changed to: {new_input_dim} (Deep Fusion)")

    def forward(self, batch):
        # 1. 文本编码器 + Pooler
        text_features = self.text_encoder(batch)
        pooled_output = self.dense(text_features)  # [Batch, Hidden_Dim]
        pooled_output = self.activation(pooled_output)

        pred_targets = pooled_output

        # 2. [关键修改] 深度融合：拼接 Text 特征和 EQ 特征
        if 'graph_embs' in batch and batch['graph_embs'] is not None and batch['graph_embs'].dim() == 2:
            eq_emb = batch['graph_embs']  # [Batch, Graph_Dim]

            # 拼接特征 [Batch, Hidden_Dim + Graph_Dim]
            fusion_output = torch.cat([pooled_output, eq_emb], dim=1)

            # 使用新的回归头进行预测 (直接输出最终预测)
            pred_targets = self.regresshead(fusion_output)

            # 返回 pred_targets 和 eq_emb (保持签名兼容)
            return pred_targets, eq_emb

            # 如果没有 EQ 嵌入，必须报错，因为模型维度已改变
        raise ValueError("Deep Fusion Model requires 'graph_embs' but none was provided or loaded.")