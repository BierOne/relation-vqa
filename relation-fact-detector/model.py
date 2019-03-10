import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

import config


class Net(nn.Module):
    def __init__(self, embeddings=None, act_fun='tanh'):
        super(Net, self).__init__()
        question_features = 2400
        vision_features = config.output_features
        share_features = 1200
        joint_features = 1200
        drop_rate = 0.5
        self.tanh = nn.Tanh()

        self.text = TextProcessor(
            embeddings=embeddings,
            embedding_features=config.text_embed_size,
            gru_features=question_features,
            drop=drop_rate
        )

        self.image_encode = FeatureMapper(
            in_features=vision_features,
            mid_features=share_features,
            out_features=joint_features,
            drop=drop_rate
        )
        self.text_encode = FeatureMapper(
            in_features=question_features,
            mid_features=share_features,
            out_features=joint_features,
            drop=drop_rate
        )
        
        self.sub_classifier = Classifier(
            in_features=joint_features,
            num_classes=2000+1,
            drop=drop_rate
        )
        self.rel_classifier = Classifier(
            in_features=joint_features,
            num_classes=256+1,
            drop=drop_rate
        )
        self.obj_classifier = Classifier(
            in_features=joint_features,
            num_classes=2000+1,
            drop=drop_rate
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, rel_sub, rel_rel, rel_obj, q_len):
        q = self.text(q, list(q_len.data))
        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)
        v = v.sum(dim=-1)

        v = self.image_encode(v)
        q = self.text_encode(q)
        h = self.tanh(v + q)

        sub_prob = self.sub_classifier(h)
        rel_prob = self.rel_classifier(h)
        obj_prob = self.obj_classifier(h)

        return sub_prob, rel_prob, obj_prob


class FeatureMapper(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(FeatureMapper, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('tanh', nn.Tanh())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class Classifier(nn.Sequential):
    def __init__(self, in_features, num_classes, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop', nn.Dropout(drop))        
        self.add_module('lin', nn.Linear(in_features, num_classes))


class TextProcessor(nn.Module):
    def __init__(self, embeddings, embedding_features, gru_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = Glove(embeddings)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.gru = nn.GRU(input_size=embedding_features,
                           hidden_size=gru_features,
                           num_layers=1,
                           batch_first=True)

        self._init_gru(self.gru.weight_ih_l0)
        self._init_gru(self.gru.weight_hh_l0)
        self.gru.bias_ih_l0.data.zero_()
        self.gru.bias_hh_l0.data.zero_()

    def _init_gru(self, weight):
        for w in weight.chunk(3, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, h = self.gru(packed)
        return h.squeeze(0)


class Glove(nn.Module):
    def __init__(self, weights, fine_tune=False):
        super(Glove, self).__init__()
        num_embeddings = weights.shape[0]
        embedding_dim = weights.shape[1]
        weights = torch.tensor(weights)

        self.embeddings = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=0)
        self.embeddings.load_state_dict({'weight': weights})
        if not fine_tune:
            self.embeddings.weight.requires_grad = False

    def forward(self, idx):
        return self.embeddings(idx)
