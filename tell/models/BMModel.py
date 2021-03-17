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
        self.use_context = use_context
        self.padding_idx = padding_value
        self.evaluate_mode = evaluate_mode
        self.sampling_topk = sampling_topk
        self.sampling_temp = sampling_temp
        self.weigh_bert = weigh_bert
        if weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history = {}

        initializer(self)

    def forward(self,  # type: ignore
                context: Dict[str, torch.LongTensor],
                labels: torch.Tensor,
                image: torch.Tensor,
                caption: Dict[str, torch.LongTensor],
                face_embeds: torch.Tensor,
                obj_embeds: torch.Tensor,
                metadata: List[Dict[str, Any]],
                names: Dict[str, torch.LongTensor] = None,
                attn_idx=None) -> Dict[str, torch.Tensor]:

        print("context: ", context)
        print("image: ", image)
        print("face_embeds: ", face_embeds)
        print("obj_embeds: ", obj_embeds)
        print("names: ", names)
        print("labels: ", labels)


        im = self.resnet(image)
        ctx = [self.roberta(p) for p in context]




        # TODO calculate attn between context paragraphs and image

        '''caption_ids, target_ids, contexts = self._forward(
            context, image, caption, face_embeds, obj_embeds)
        decoder_out = self.decoder(caption, contexts)'''

        output_dict = {
            'loss': 0
        }

        # During evaluation...
        if not self.training and self.evaluate_mode:
            pass

        self.n_batches += 1

        return output_dict
