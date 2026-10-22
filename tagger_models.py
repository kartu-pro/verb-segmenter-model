"""
tagger_models.py
----------------
PyTorch model architectures for Sequence Tagging (Standard BiLSTM & BiLSTM-CRF).
"""

import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMMetadataTagger(nn.Module):
    """BiLSTM Model with metadata feature concatenation."""

    def __init__(self, vocabs: dict, char_dim: int = 64, meta_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.char_embed = nn.Embedding(len(vocabs["char2id"]), char_dim, padding_idx=0)
        self.screeve_embed = nn.Embedding(len(vocabs["screeve2id"]), meta_dim)
        self.subject_embed = nn.Embedding(len(vocabs["subject2id"]), meta_dim)
        self.class_embed = nn.Embedding(len(vocabs["class2id"]), meta_dim)

        input_dim = char_dim + (meta_dim * 3)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, len(vocabs["label2id"]))

    def forward(self, tokens, screeve_id, subject_id, class_id):
        batch_size, seq_len = tokens.shape
        c_emb = self.char_embed(tokens)
        s_emb = self.screeve_embed(screeve_id).unsqueeze(1).expand(-1, seq_len, -1)
        subj_emb = self.subject_embed(subject_id).unsqueeze(1).expand(-1, seq_len, -1)
        cls_emb = self.class_embed(class_id).unsqueeze(1).expand(-1, seq_len, -1)

        combined = torch.cat([c_emb, s_emb, subj_emb, cls_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        logits = self.fc(lstm_out)
        return logits


class BiLSTMMetadataTaggerWithCRF(nn.Module):
    """BiLSTM Model with metadata feature concatenation and CRF output layer."""

    def __init__(self, vocabs: dict, char_dim: int = 64, meta_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.char_embed = nn.Embedding(len(vocabs["char2id"]), char_dim, padding_idx=0)
        self.screeve_embed = nn.Embedding(len(vocabs["screeve2id"]), meta_dim)
        self.subject_embed = nn.Embedding(len(vocabs["subject2id"]), meta_dim)
        self.class_embed = nn.Embedding(len(vocabs["class2id"]), meta_dim)

        input_dim = char_dim + (meta_dim * 3)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, len(vocabs["label2id"]))
        self.crf = CRF(len(vocabs["label2id"]), batch_first=True)

    def forward(self, tokens, screeve_id, subject_id, class_id, labels=None, mask=None):
        batch_size, seq_len = tokens.shape
        c_emb = self.char_embed(tokens)
        s_emb = self.screeve_embed(screeve_id).unsqueeze(1).expand(-1, seq_len, -1)
        subj_emb = self.subject_embed(subject_id).unsqueeze(1).expand(-1, seq_len, -1)
        cls_emb = self.class_embed(class_id).unsqueeze(1).expand(-1, seq_len, -1)

        combined = torch.cat([c_emb, s_emb, subj_emb, cls_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        emissions = self.fc(lstm_out)

        if mask is None:
            mask = (tokens != 0)

        if labels is not None:
            return -self.crf(emissions, labels, mask=mask, reduction='mean')
        else:
            return self.crf.decode(emissions, mask=mask)