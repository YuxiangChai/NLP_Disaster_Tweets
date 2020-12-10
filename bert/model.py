from transformers import BertModel
import torch.nn as nn


class Bert(nn.Module):
    def __init__(self, n_classes):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        result = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = result[1]
        output = self.dropout(pooled)
        return self.out(output)