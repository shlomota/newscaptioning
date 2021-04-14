from allennlp.models.model import Model
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.initializers import InitializerApplicator
import numpy as np

import copy
import math
import re
from collections import defaultdict

from overrides import overrides
from pycocoevalcap.bleu.bleu_scorer import BleuScorer

from tell.modules.criteria import Criterion
from .resnet import resnet152


def split_list(li, val):
    result = [[]]
    i = 0
    while i < len(li):
        if li[i] == val:
            result += [[]]
            i += 1
            while i < len(li) and li[i] == val:
                i += 1
        else:
            result[-1].append(li[i])
            i += 1
    return result


@Model.register("BMRelModel")
class BMRelModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 criterion: Criterion,
                 evaluate_mode: bool = False,
                 attention_dim: int = 1024,
                 hidden_size: int = 1024,
                 dropout: float = 0.1,
                 vocab_size: int = 50264,
                 model_name: str = 'roberta-base',
                 namespace: str = 'bpe',
                 index: str = 'roberta',
                 padding_value: int = 1,
                 use_context: bool = True,
                 sampling_topk: int = 1,
                 sampling_temp: float = 1.0,
                 weigh_bert: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        # self.criterion = criterion
        self.criterion = nn.BCELoss()

        self.index = index
        self.namespace = namespace
        self.resnet = resnet152()
        self.roberta = torch.hub.load(
            'pytorch/fairseq:2f7e3f3323', 'roberta.large')
        self.roberta.eval()
        self.use_context = use_context
        self.padding_idx = padding_value
        self.evaluate_mode = evaluate_mode
        self.sampling_topk = sampling_topk
        self.sampling_temp = sampling_temp
        self.weigh_bert = weigh_bert

        self.loss_func = nn.MSELoss()

        self.conv = nn.Conv2d(2048, 512, 7)
        self.linear = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        if weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history = {}

        initializer(self)

    def forward(self,  # type: ignore
                context: List[Dict[str, torch.LongTensor]],
                label: torch.Tensor,
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                face_embeds: torch.Tensor,
                obj_embeds: torch.Tensor,
                metadata: List[Dict[str, Any]],
                names: Dict[str, torch.LongTensor] = None,
                attn_idx=None) -> Dict[str, torch.Tensor]:

        label = label.squeeze()
        im = self.resnet(image).detach()
        conv = self.conv(im)
        if conv.shape[0] == 1:
            conv = conv[0].squeeze().unsqueeze(0)
        else:
            conv = conv.squeeze()

        # mask = context['roberta_copy_masks']  # [B,K,N]
        # B, N = mask.shape[0], mask.shape[2]
        # v = torch.zeros((N, 1), dtype=int)  # [N,1]
        # v[0] = 1  # [ 1, 0, ... 0 ]
        # v = v.expand((B, N, 1))  # [B,N,1]
        # mask = torch.bmm(mask, v).squeeze(-1).bool()  # [B,K]

        im_vec = F.relu(conv)  # [ 512 ]
        c = context['roberta']  # [B, K, N]
        mask = context["roberta_copy_masks"]  # [B, K, N]
        hiddens = torch.stack([self.roberta.extract_features(p).detach() for p in c])  # [B, K, N, 1024]
        # cshape = c.shape
        # c = c.view(cshape[0] * cshape[1], -1)  # [BK, N]
        # hiddens = self.roberta.extract_features(c).detach().view(cshape+(1024,))  # [B, K, N, 1024]
        '''torch.save(c, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/c.pt')
        torch.save(hiddens, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/h.pt')
        torch.save(mask, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/mask.pt')
        raise Exception("whatever")'''

        # TODO: check if we want to address sentence length and calculate mean w.r.t that
        # h = torch.mean(hiddens, dim=2)  # [B, K, 1024]
        sums = torch.sum(hiddens, dim=2)
        h = sums / np.tile(np.argmin(mask, axis=-1)[:, :, np.newaxis], sums.shape[-1]).astype(np.float32)
        # nan -> 0 before passing through layers, then we mask these paragraphs out anyway
        # h = torch.nan_to_num(h) # doesn't exist in torch 1.5.1
        h[torch.isnan(h)] = 0
        text_vec = F.relu(self.linear(h))  # [B, K, 512]
        score = torch.bmm(text_vec, im_vec.unsqueeze(-1)).squeeze(-1)  # [B, K, 512] bmm [B, 512, 1] . s = [B,K]
        single_value_score = torch.softmax(score, dim=1)[:, 1]

        loss = self.criterion(single_value_score, label)

        output_dict = {
            'score0': score[:, 0],
            'score1': score[:, 1],
            'loss': loss,
            # 'probs': score
        }

        self.n_batches += 1

        strloss = f'{self.n_batches}:{loss}'
        print(strloss)

        open('/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/BMRel.log', 'a').write(strloss + '\n')

        # raise Exception("bla")

        return output_dict
