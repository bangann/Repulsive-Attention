import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *


class SentenceEmbedding(nn.Module):
    """
    Prepare and encode sentence embeddings
    """

    def __init__(self, config):
        super(SentenceEmbedding, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.embed_size, config.embed_dim)
        self.encoder = Encoder(config)

    def forward(self, input_sentence):
        sentence = self.word_embedding(input_sentence)
        embedding, attention = self.encoder(input_sentence, sentence)
        return embedding, attention


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=1,
                           bidirectional=True)

        # self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.selfattention = SelfAttention(config)

    def forward(self, inputs_org, inputs):
        batch_size = inputs.size()[1]
        # x = F.dropout(inputs, self.config.dropout, training=self.training)
        x = inputs
        h_0 = c_0 = Variable(inputs.data.new(2, batch_size, self.config.embed_dim).zero_())
        embedding = self.rnn(x, (h_0, c_0))[0]

        # # Max pooling
        # emb = self.max_pool(embedding.permute(1, 2, 0))
        # emb = emb.squeeze(2)

        # self attention
        embedding = embedding.permute(1, 0, 2)
        emb, attention = self.selfattention(inputs_org, embedding)
        # emb = emb.view(emb.size(0), -1)

        return emb, attention


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.config = config
        self.attention_hops = config.attention_hops
        self.attention_unit = config.attention_unit
        # self.drop = nn.Dropout(config.dropout)
        self.ws1 = nn.Linear(config.hidden_dim * 2, config.attention_unit, bias=False)
        self.ws2 = nn.Linear(config.attention_unit, config.attention_hops, bias=False)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax()

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, outp):
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.contiguous().view(-1, size[2])  # [bsz*len, nhid*2]

        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for _ in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        # hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        hbar = self.tanh(self.ws1(compressed_embeddings))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        penalized_alphas = alphas + (
                -100000 * (concatenated_inp == 1).float())

        # [bsz, hop, len] + [bsz, hop, len]
        alphas = F.softmax(penalized_alphas.view(-1, size[1]), dim=1)  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]

        return torch.bmm(alphas, outp), alphas


