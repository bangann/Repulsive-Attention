import torch
import torch.nn as nn
import numpy as np
from embedding_svgd import SentenceEmbedding


class FCClassifier(nn.Module):

    def __init__(self, config):
        super(FCClassifier, self).__init__()
        self.config = config
        self.dropout = config.dropout
        if config.activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif config.activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.seq_in_size = config.attention_hops * (2 * config.hidden_dim)
        self.fc_dim = config.fc_dim
        self.out_dim = config.out_dim

        self.mlp = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.seq_in_size, self.fc_dim),
            self.activation,
            nn.Dropout(p=self.dropout),
            # nn.Linear(self.fc_dim, self.fc_dim),
            # self.activation,
            nn.Linear(self.fc_dim, self.out_dim)
        )

    def forward(self, sentence_emb):
        output = self.mlp(sentence_emb)
        return output


class Classification(nn.Module):
    def __init__(self, config):
        super(Classification, self).__init__()
        self.config = config
        self.sentence_embedding = SentenceEmbedding(config)
        if config.corpus == 'snli':
            self.gatedencoder3d = GatedEncoder3D(config)
        self.classifier = FCClassifier(config)

    def forward(self, batch):
        if self.config.corpus in ('yelp', 'age'):
            sent_emb, attention = self.sentence_embedding(batch.input)

        if self.config.corpus == 'snli':
            prem, attention = self.sentence_embedding(batch.premise)
            hypo, _ = self.sentence_embedding(batch.hypothesis)
            sent_emb = self.gatedencoder3d(prem, hypo)

        sent_emb = sent_emb.contiguous().view(sent_emb.size(0), -1)
        answer = self.classifier(sent_emb)

        return answer, attention


class GatedEncoder3D(nn.Module):
    def __init__(self, config):
        super(GatedEncoder3D, self).__init__()
        self.config = config
        self.Wfh = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(config.attention_hops, 2 * config.hidden_dim, 2 * config.hidden_dim).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.Wfp = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(config.attention_hops, 2 * config.hidden_dim, 2 * config.hidden_dim).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, prem, hypo):
        prem = torch.einsum('abc,bcd->abd', [prem, self.Wfp])
        hypo = torch.einsum('abc,bcd->abd', [hypo, self.Wfh])
        sent_emb = prem * hypo
        return sent_emb
