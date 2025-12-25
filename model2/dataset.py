import torch


class ClipDataset(torch.utils.data.Dataset):

    def __init__(self, texts, targets, graph_emb, tokenizer, seq_len=512):
        self.texts = texts
        self.targets = targets
        self.graph_emb = graph_emb
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        """Returns the length of dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        For a given item index, return a dictionary of encoded information
        """
        text = str(self.texts[idx])

        tokenized = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        return {"input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
                "target": torch.tensor(self.targets[idx], dtype=torch.float),
                "graph_embed": torch.tensor(self.graph_emb[idx], dtype=torch.float),
                }


class RegressionDataset(torch.utils.data.Dataset):

    # [修正] 增加 graph_embs 参数
    def __init__(self, texts, targets, tokenizer, seq_len=512, graph_embs=None):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # 存储 graph_embs 和状态
        self.graph_embs = graph_embs
        self.has_graph_embs = self.graph_embs is not None

    def __len__(self):
        """Returns the length of dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        For a given item index, return a dictionary of encoded information
        """
        text = str(self.texts[idx])

        tokenized = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        data = {"input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
                "target": torch.tensor(self.targets[idx], dtype=torch.float),
                }

        # [关键修复] 只有当 graph_embs 存在时，才将该键加入到返回的字典中。
        if self.has_graph_embs:
            data["graph_embs"] = torch.tensor(self.graph_embs[idx], dtype=torch.float)

        return data