from common.schema.input_features import InputFeatures
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, features: list[InputFeatures]):
        self.input_ids_list = [f.input_ids for f in features]
        self.attention_mask_list = [f.attention_mask for f in features]
        self.labels_list = [f.label for f in features if isinstance(f.label, int) or isinstance(f.label, list)]  # 不getattr则返回[None]

    def __len__(self):
        return len(self.input_ids_list)

    # 只有返回dict形式data collator才生效，tuple形式不生效
    def __getitem__(self, item):
        if self.labels_list:
            return {"input_ids": self.input_ids_list[item],
                    "attention_mask": self.attention_mask_list[item],
                    "labels": self.labels_list[item]}
        else:
            return {"input_ids": self.input_ids_list[item], "attention_mask": self.attention_mask_list[item]}
