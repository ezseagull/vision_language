import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu

class Bert(nn.Module):
    """Finetunable language backbone shared by CRA and TempCLIP.

    Q-Bound V3.8-Minimal keeps the exact pretrained model and embedding path.
    PAD tokens are excluded from transformer self-attention using the active
    tokenizer's pad_token_id (RoBERTa commonly uses 1, NOT an assumed 0).

    CRA's learned question/answer pooling lives in
    modules.m_cra.only_trans.LanModel, NOT in the legacy LanModel below.
    """

    def __init__(self, tokenizer, lan):
        super(Bert, self).__init__()
        
        if lan == 'DistilBERT':
            from transformers import DistilBertModel, DistilBertConfig
            config = DistilBertConfig.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        elif lan == 'BERT':
            from transformers import BertModel, BertConfig
            config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
            self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        elif lan == 'RoBERTa':
            from transformers import RobertaModel, RobertaConfig
            config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
            self.bert = RobertaModel.from_pretrained("roberta-base", config=config)
        elif lan == 'DeBERTa':
            from transformers import DebertaConfig, DebertaModel
            config = DebertaConfig.from_pretrained("microsoft/deberta-base", output_hidden_states=True)
            self.bert = DebertaModel.from_pretrained("microsoft/deberta-base", config=config)
        else:
            raise ValueError(
                "Unsupported language backbone {!r}; choose DistilBERT, BERT, "
                "RoBERTa or DeBERTa.".format(lan)
            )

        self.tokenizer = tokenizer
        
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False
        
    def get_attention_mask(self, tokens):
        """Boolean [B,L] mask aligned with tokenizer-aware candidate padding.

        No trainable parameters, no change to embeddings, and no hard-coded
        RoBERTa/BERT PAD ID. Mask validation is deliberately strict so an
        invalid batch fails before an expensive backbone forward.
        """
        if not torch.is_tensor(tokens) or tokens.ndim != 2:
            raise ValueError(
                "Bert expects integer token ids [B,L], got "
                f"{type(tokens)!r} / {getattr(tokens, 'shape', None)}."
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("Bert token IDs must use int32 or int64 dtype.")
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id for language masking.")
        mask = tokens.ne(int(pad_token_id))
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Bert received an empty or all-PAD sequence.")
        return mask

    def forward(self, tokens): #, seq_len, seg_feats, seg_num):
        # Retain the original FLOAT attention_mask for checkpoint/runtime
        # compatibility. The original code already supplied this mask;
        # only its validation has been made explicit.
        attention_mask = self.get_attention_mask(tokens).float().to(tokens.device)
        outs = self.bert(tokens, attention_mask=attention_mask)
        embds = outs[0]
        return embds


class Sentence_Maxpool(nn.Module):
    """ Utilitary for the answer module """

    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x_in):
        x = self.fc(x_in)
        x = torch.max(x, dim=1)[0]
        if self.relu:
            x = F.relu(x)
        return x


class FFN(nn.Module):
    def __init__(self, word_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        activation = "gelu"
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=word_dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class LanModel(nn.Module):
    """
    Language embedding module
    """

    def __init__(self, tokenizer, lan, word_dim=768, out_dim=512):
        super(LanModel, self).__init__()

        self.bert = Bert(tokenizer, lan)
        self.linear_text = nn.Linear(word_dim, out_dim)

        # self.linear_text = FFN(word_dim, out_dim, out_dim)

    def _check_device(self, module, x):
        for name, param in module.named_parameters():
            if param.device != x.device:
                print(f"Parameter {name} is on device {param.device}, expected {x.device}")
        
    def forward(self, answer):
        self._check_device(self.bert, answer)
        
        if len(answer.shape) == 3:
            #multi-choice
            bs, nans, lans = answer.shape
            answer = answer.view(bs * nans, lans)
            answer = self.bert(answer)
            answer = self.linear_text(answer)
            answer_g = answer.mean(dim=1)
            # answer_g = answer[:, 0, :]
            answer_g = answer_g.view(bs, nans, -1)
            
            return answer_g, answer.view(bs, nans, lans, -1)
        else:
            answer = self.bert(answer)
            answer = self.linear_text(answer)
            answer_g = answer.mean(dim=1)
            # answer_g = answer[:, 0, :]
        
            return answer_g, answer
