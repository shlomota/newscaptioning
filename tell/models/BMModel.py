from allennlp.models.model import Model
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.initializers import InitializerApplicator

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

@Model.register("BMModel")
class BMModel(Model):
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
        self.criterion = criterion

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
        # self.linear = nn.Linear(2048, 512)
        self.linear = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        if weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history = {}

        self.dbr = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/dbr/"

        initializer(self)

    def forward(self,  # type: ignore
                aid: List[str],
                split: List[str],
                index: torch.Tensor,
                label: torch.Tensor,
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                # face_embeds: torch.Tensor,
                # obj_embeds: torch.Tensor,
                # metadata: List[Dict[str, Any]],
                # names: Dict[str, torch.LongTensor] = None,
                attn_idx=None) -> Dict[str, torch.Tensor]:

        # aid = [hex(int("".join(map(str, map(int, i)))))[2:] for i in aid]  # [B]

        split = split[0]

        dbrf = ''
        a = 1
        if split == 'test' or split == 'valid':
            dbrf = split

        label = label.squeeze()
        im = self.resnet(image).detach()

        conv = self.conv(im)
        if conv.shape[0] == 1:
            conv = conv[0].squeeze().unsqueeze(0)
        else:
            conv = conv.squeeze()

        im_vec = self.relu(conv)

        hiddens = [torch.load(f"{self.dbr}{dbrf}/{i}")[index[i]] for i in aid]
        masks = [torch.load(f"{self.dbr}{dbrf}/{i}m")[index[i]] for i in aid]
        m = [torch.add(i.unsqueeze(-1).expand(*i.shape, 1024), 1) for i in masks]
        # hiddens = self.roberta.extract_features(context["roberta"]).detach()
        # using only first and last hidden because size can change
        # h = torch.cat([hiddens[:,0,:], hiddens[:,-1,:]], dim=-1)
        h = torch.mean(hiddens, dim=1)
        text_vec = self.relu(self.linear(h))

        # TODO: use tensors and correct code
        # scores = torch.tensor([im @ p for p in split_context])
        score = torch.bmm(text_vec.unsqueeze(1), im_vec.unsqueeze(-1)).squeeze()
        # sm_scores = nn.Softmax()(scores) #use torch nn

        loss = self.loss_func(score, label)
        # loss = nn.CrossEntropyLoss(scores, labels)



        '''caption_ids, target_ids, contexts = self._forward(
            context, image, caption, face_embeds, obj_embeds)
        decoder_out = self.decoder(caption, contexts)'''

        output_dict = {
            'loss': loss,
            'probs': score
        }

        # During evaluation...
        if not self.training and self.evaluate_mode:
            pass

        self.n_batches += 1

        return output_dict