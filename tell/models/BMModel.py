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

        initializer(self)

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                label: torch.Tensor,
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                face_embeds: torch.Tensor,
                obj_embeds: torch.Tensor,
                metadata: List[Dict[str, Any]],
                names: Dict[str, torch.LongTensor] = None,
                attn_idx=None) -> Dict[str, torch.Tensor]:

        # stage 1: use only resnet of image and roberta of text (and linear layers)
        im = self.resnet(image).detach()
        conv = self.conv(im)
        if conv.shape[0] == 1:
            conv = conv[0].squeeze().unsqueeze(0)
        else:
            conv = conv.squeeze()
        im_vec = F.relu(conv)

        hiddens = self.roberta.extract_features(context["roberta"]).detach()
        mask = context["roberta_copy_masks"]  # [B, N]
        # h = torch.mean(hiddens, dim=1)
        sums = torch.sum(hiddens, dim=1)
        h = sums / np.tile(np.argmin(mask, axis=-1)[:, :, np.newaxis], sums.shape[-1]).astype(np.float32)
        h[torch.isnan(h)] = 0
        h[torch.isinf(h)] = 1e15
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


