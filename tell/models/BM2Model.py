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


@Model.register("BM2Model")
class BM2Model(Model):
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
                 arch: int = 0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        # self.criterion = criterion
        self.criterion = nn.KLDivLoss()

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

        self.arch = arch
        print("arch", self.arch)

        if self.arch in [0]:
            self.linear = nn.Linear(1024, 512)
        else:
            self.linear = nn.Linear(1024, 768)
            self.linear2 = nn.Linear(768, 512)

        if weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)

        self.n_batches = 0
        self.n_samples = 0
        self.sample_history = {}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dbr = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/dbr/"

        initializer(self)

    def forward(self,  # type: ignore
                aid: List[str],
                split: List[str],
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
        if split == 'test' or split == 'valid':
            dbrf = split

        label = label.squeeze()
        im = self.resnet(image).detach()
        conv = self.conv(im)
        if conv.shape[0] == 1:
            conv = conv[0].squeeze().unsqueeze(0)
        else:
            conv = conv.squeeze()

        # mask = context['roberta_copy_masks']  # [B,K,N]
        '''B, N = mask.shape[0], mask.shape[2]
        print("B", B)
        v = torch.zeros((N, 1), dtype=int)  # [N,1]
        v[0] = 1  # [ 1, 0, ... 0 ]
        v = v.expand((B, N, 1))  # [B,N,1]
        mask = torch.bmm(mask, v).squeeze(-1).bool()  # [B,K]'''

        if self.arch in [2]:
            im_vec = conv  # [ 512 ]
        else:
            im_vec = F.relu(conv)  # [ 512 ]

        # c = context['roberta']  # [B, K, N]
        # hiddens = torch.stack([self.roberta.extract_features(p).detach() for p in c])  # [B, K, N, 1024]

        c = [torch.load(f"{self.dbr}{dbrf}/{i}") for i in aid]
        m = [torch.load(f"{self.dbr}{dbrf}/{i}m") for i in aid]
        m = [torch.add(i.unsqueeze(-1).expand(*i.shape, 1024), 1) for i in m]
        #m = [i.bool() for i in m]

        c = [torch.sum(cv*m[ci], dim=1)/torch.sum(m[ci], dim=1) for ci, cv in enumerate(c)]

        cshapes = torch.tensor([cv.shape[0] for cv in c])
        c = torch.nn.utils.rnn.pad_sequence(c, batch_first=True).to(self.device)  # [B, N, 1024]

        print("asdf")

        #torch.save(c, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/c.pt')

        #c = (torch.arange(c.size(1)) < cshapes[..., None]).unsqueeze(-1).repeat(1, 1, 1024) * c

        # cshape = c.shape
        # c = c.view(cshape[0] * cshape[1], -1)  # [BK, N]
        # hiddens = self.roberta.extract_features(c).detach().view(cshape+(1024,))  # [B, K, N, 1024]
        '''torch.save(c, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/c.pt')
        torch.save(hiddens, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/h.pt')
        torch.save(mask, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/mask.pt')
        raise Exception("whatever")'''

        # TODO: check if we want to address sentence length and calculate mean w.r.t that
        #h = torch.mean(hiddens, dim=2)  # [B, K, 1024]
        # nan -> 0 before passing through layers, then we mask these paragraphs out anyway
        # h = torch.nan_to_num(h) # doesn't exist in torch 1.5.1
        #h[torch.isnan(h)] = 0

        text_vec = F.relu(self.linear(c))

        if self.arch in [1, 2]:
            text_vec = self.linear2(text_vec)

        if self.arch in [1]:
            text_vec = F.relu(text_vec)

        #torch.save(text_vec, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/tv.pt')

        # text_vec [B, N, 512]

        score = torch.bmm(text_vec, im_vec.unsqueeze(-1)).squeeze(-1)  # [B, N, 512] bmm [B, 512, 1] . s = [B, N]
        score_mask = ~(torch.arange(score.size(1)) < cshapes[..., None])  # [B, N] pad out
        score[score_mask] = float("-inf")

        # torch.save(mask, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/mask.pt')
        # torch.save(score, '/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/score.pt')

        score = F.log_softmax(score)

        loss = self.criterion(score, label)

        output_dict = {
            'loss': loss,
            'probs': score
        }

        # During evaluation...
        # if not self.training and self.evaluate_mode:
        #    pass

        if split == 'train':
            self.n_batches += 1
            strloss = f'{self.n_batches}:{loss}'
            print(strloss)

            #open('/a/home/cc/students/cs/shlomotannor/nlp_course/newscaptioning/BMM.log', 'a').write(strloss + '\n')

        return output_dict