from argparse import ArgumentParser
from json import decoder
from logging import debug
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import wandb
import numpy as np
# Hide lines above until Lab 5

from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup

from functools import partial

import random

def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0  # 只优化id为1～8的token
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

# class AMSoftmax(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_classes=10,
#                  m=0.35,
#                  s=30):
#         super(AMSoftmax, self).__init__()
#         self.m = m
#         self.s = s
#         self.in_feats = in_feats
#         self.W = torch.nn.Linear(in_feats, n_classes)
#         # self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
#         self.ce = nn.CrossEntropyLoss()
#         # nn.init.xavier_normal_(self.W, gain=1)

#     def forward(self, x, lb):
#         assert x.size()[0] == lb.size()[0]
#         assert x.size()[1] == self.in_feats
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         x_norm = torch.div(x, x_norm)
#         w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
#         w_norm = torch.div(self.W, w_norm)
#         costh = torch.mm(x_norm, w_norm)
#         # print(x_norm.shape, w_norm.shape, costh.shape)
#         lb_view = lb.view(-1, 1)
#         if lb_view.is_cuda: lb_view = lb_view.cpu()
#         delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
#         if x.is_cuda: delt_costh = delt_costh.cuda()
#         costh_m = costh - delt_costh
#         costh_m_s = self.s * costh_m
#         loss = self.ce(costh_m_s, lb)
#         return loss, costh_m_s

# class AMSoftmax(nn.Module):

#     def __init__(self, in_features, out_features, s=30.0, m=0.35):
#         '''
#         AM Softmax Loss
#         '''
#         super().__init__()
#         self.s = s
#         self.m = m
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = nn.Linear(in_features, out_features, bias=False)

#     def forward(self, x, labels):
#         '''
#         input shape (N, in_features)
#         '''
#         assert len(x) == len(labels)
#         assert torch.min(labels) >= 0
#         assert torch.max(labels) < self.out_features
#         for W in self.fc.parameters():
#             W = F.normalize(W, dim=1)

#         x = F.normalize(x, dim=1)

#         wf = self.fc(x)
#         numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
#         excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
#         denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
#         L = numerator - torch.log(denominator)
#         return -torch.mean(L)


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # self.loss_fn = AMSoftmax(self.model.config.hidden_size, num_relation)
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.tokenizer = tokenizer
    
        self._init_label_word()
        
        # with torch.no_grad():
        #     self.loss_fn.fc.weight = nn.Parameter(self.model.get_output_embeddings().weight[self.label_st_id:self.label_st_id+num_relation])
            # self.loss_fn.fc.bias = nn.Parameter(self.model.get_output_embeddings().bias[self.label_st_id:self.label_st_id+num_relation])

    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        
        num_labels = len(label_word_idx)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]
            
            # for abaltion study
            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]
            
            if self.args.init_type_words:
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        
        self.word2label = continous_label_word # a continous list
            
                
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids, labels, so = batch  # 证明我们之前对于batch的拆解是正确的
        '''
        input_ids: （batchsize，seq_length）
        attention_mask: （batchsize，seq_length）
        token_type_ids: （batchsize，seq_length）
        labels:（batchsize）
        so：？？？
        '''
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits  # [batch_size, seq_len, vocab_size]，MaskedLMOutput(loss=None, logits=tensor([[[-0.0696, -0.8332, -0.2249,  ..., -0.6001,  0.3200,  0.0798],         [-0.0738, -0.8464,  0.2009,  ..., -0.7553,  0.0935,  0.0346],         [-0.0810, -0.7307,  0.1313,  ..., -0.4871,  0.2473,  0.0191],         ...,         [-0.0698, -0.7097,  0.2113,  ..., -0.6917,  0.0658, -0.1001],         [-0.0812, -0.6459,  0.0600,  ..., -0.5740,  0.0242, -0.0878],         [-0.0714, -0.6099,  0.0329,  ..., -0.7716,  0.1355, -0.0215]]],       device='cuda:0', grad_fn=<ViewBackward0>), hidden_states=(tensor([[[ 1.1301, -0.2193, -0.3826,  ...,  0.9539, -0.9765,  1.0902],         [ 0.4858,  1.3283,  0.1391,  ...,  0.0780, -1.5029,  0.6699],         [-1.0316, -1.3581, -0.5338,  ...,  0.0000, -1.4681,  1.5334],         ...,         [ 0.0570,  0.1846,  0.0842,  ...,  0.8740, -1.9086,  0.4485],         [ 0.0570,  0.1846,  0.0842,  ...,  0.8740, -1.9086,  0.4485],         [ 0.0570,  0.1846,  0.0842,  ...,  0.8740, -1.9086,  0.0000]]],       device='cuda:0', grad_fn=<NativeDropoutBackward0>), tensor([[[ 1.0116, -0.7400, -0.0277,  ...,  1.1618, -0.3470,  0.8387],         [ 0.4280,  0.9437,  0.6123,  ...,  1.0500, -0.7346,  0.5871],         [-1.1242, -1.3320, -0.4791,  ..., -0.0760, -0.8483,  1.4023],         ...,         [ 0.0194,  0.1672,  0.4876,  ...,  1.7718, -1.1094,  0.3408],         [ 0.1624, -0.0087,  0.3020,  ...,  1.7354, -1.2464,  0.3634],         [ 0.0266,  0.0451, -0.0252,  ...,  1.0945, -1.0042, -0.2458]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 1.1870, -0.6340, -0.0026,  ...,  1.7075,  0.2315,  0.2791],         [ 1.0255,  0.1026,  0.4015,  ...,  1.9415, -0.6765,  0.8880],         [-1.3961, -1.1775, -0.4087,  ...,  0.3719, -0.8951,  1.1611],         ...,         [ 0.2285, -0.6897,  0.5579,  ...,  2.0589, -0.7603,  0.3311],         [ 0.3829, -0.6500,  0.2488,  ...,  2.4267, -0.9171,  0.2179],         [-0.0614, -0.7702,  0.0205,  ...,  1.7995, -0.6389, -0.4973]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 0.9458, -0.5010,  0.2106,  ...,  2.7974,  0.7517,  0.7232],         [ 0.8409, -0.2768,  0.5790,  ...,  2.1081, -0.1994,  0.8355],         [-1.2890, -1.1979, -0.1041,  ...,  1.5679, -0.1431,  0.5866],         ...,         [-0.1833, -1.4371,  0.9174,  ...,  2.2119, -0.3806,  0.4292],         [ 0.0961, -1.4828,  0.4890,  ...,  2.9521, -0.0531,  0.4112],         [-0.3871, -1.1779, -0.0443,  ...,  1.9872,  0.1676, -0.4287]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 1.1577, -0.0334,  0.0419,  ...,  2.8417,  0.8165,  0.7990],         [ 0.5000, -0.6324,  0.6553,  ...,  1.6932,  0.2186,  0.9562],         [-1.4103, -1.4743, -0.0316,  ...,  1.6573,  0.2453,  0.3279],         ...,         [-0.3377, -1.3992,  0.6531,  ...,  2.1177,  0.2699,  0.2504],         [-0.2162, -1.6523,  0.2051,  ...,  2.9882,  0.3147,  0.4360],         [-0.7518, -1.3907,  0.0877,  ...,  1.9529,  0.8557, -0.3825]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 2.0567, -0.5220,  0.0596,  ...,  3.1570,  0.6912,  1.0035],         [ 1.4669, -1.5104,  0.7238,  ...,  1.4239, -0.0042,  1.1967],         [-0.6369, -1.7288, -0.3405,  ...,  1.3848,  0.3379,  0.3804],         ...,         [ 0.6025, -1.9860,  0.4809,  ...,  1.7299,  0.1783,  0.2426],         [ 0.5614, -2.1997,  0.1840,  ...,  2.5364, -0.0429,  0.2101],         [-0.0465, -1.5755, -0.1254,  ...,  1.4797,  0.6938, -0.3348]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 2.6682, -0.8503,  0.7542,  ...,  2.6875,  0.9322,  1.3257],         [ 1.6133, -1.7742,  1.6678,  ...,  1.2993,  0.2167,  0.9181],         [ 0.2199, -2.1729,  0.4537,  ...,  1.4575,  0.5353,  0.5154],         ...,         [ 1.4072, -2.0787,  0.7408,  ...,  1.8613,  0.5408,  0.6094],         [ 0.9775, -2.2061,  0.9519,  ...,  2.3728,  0.2613,  0.4558],         [-0.0394, -1.6517,  0.4634,  ...,  1.7791,  0.4882, -0.1022]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 2.2485, -0.8819,  0.4605,  ...,  2.1923,  1.2429,  0.9886],         [ 1.4276, -1.8577,  0.9270,  ...,  1.1063,  1.2446,  1.0395],         [ 0.1042, -2.5375,  0.1405,  ...,  1.5376,  1.1384,  0.6005],         ...,         [ 1.3668, -2.2975,  0.0333,  ...,  1.5102,  1.0350,  0.5329],         [ 0.7671, -2.6348,  0.1871,  ...,  1.8223,  0.7634,  0.5776],         [ 0.2359, -2.2552, -0.1612,  ...,  1.5090,  0.6319,  0.2484]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 2.3467, -1.2682,  0.8023,  ...,  1.4901,  0.7963,  1.3030],         [ 1.7206, -1.4130,  0.7902,  ...,  1.1388,  0.9615,  1.0292],         [ 0.3466, -2.7555, -0.0180,  ...,  1.8056,  0.9495,  1.1019],         ...,         [ 1.0659, -2.4597, -0.0285,  ...,  1.5449,  0.6879,  0.8096],         [ 1.0864, -2.9219,  0.1931,  ...,  2.0038,  0.4529,  0.8707],         [ 0.4193, -2.4531, -0.2118,  ...,  1.6117,  0.3180,  0.6740]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 1.7279, -1.1916,  0.3418,  ...,  1.9500,  0.5708,  1.4367],         [ 1.3319, -1.2226,  0.6417,  ...,  1.5956,  0.8412,  0.8954],         [ 0.7222, -2.2341,  0.0131,  ...,  1.3453,  0.4395,  1.2136],         ...,         [ 1.2285, -2.0251, -0.1062,  ...,  1.6589,  0.6505,  0.7897],         [ 1.1684, -2.4594, -0.0627,  ...,  2.0490,  0.1113,  0.9148],         [ 0.6804, -2.0337, -0.3286,  ...,  1.8290, -0.0145,  0.6378]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 1.2706, -0.7127,  0.4632,  ...,  2.4318,  0.1804,  0.6502],         [ 1.5823, -0.9240,  0.5774,  ...,  1.8245,  0.6022, -0.3147],         [ 1.2220, -1.8555,  0.1274,  ...,  1.6426,  0.2436,  0.0805],         ...,         [ 1.7588, -1.6551, -0.6085,  ...,  1.7983,  0.2179, -0.3634],         [ 1.4729, -2.0162, -0.0210,  ...,  2.3071, -0.0207,  0.2602],         [ 1.5349, -1.5890, -0.0320,  ...,  1.9795,  0.0948, -0.4707]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 0.6381, -0.8477,  1.0872,  ...,  1.5120,  0.4253,  0.2356],         [ 0.7148, -0.9343,  0.9321,  ...,  0.9608,  0.4220, -0.7096],         [ 0.5213, -1.5522,  0.7022,  ...,  1.0253,  0.3110, -0.0675],         ...,         [ 0.9226, -1.3541,  0.1220,  ...,  1.0889,  0.3842, -0.4636],         [ 0.5756, -1.8199,  0.6014,  ...,  1.2788,  0.2536, -0.2004],         [ 0.8414, -1.6249,  0.4946,  ...,  1.4113,  0.4365, -0.5932]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>), tensor([[[ 0.1379, -0.9885,  0.8207,  ...,  1.6779,  0.7580,  0.7801],         [ 0.1710, -1.2076,  1.0342,  ...,  1.3896,  0.6267,  0.1699],         [ 0.0739, -1.5097,  0.6652,  ...,  1.3992,  0.7127,  0.5576],         ...,         [ 0.5178, -1.3009,  0.0591,  ...,  1.4608,  0.3629,  0.2541],         [ 0.3029, -1.6019,  0.7178,  ...,  1.2213,  0.4982,  0.6291],         [ 0.4637, -1.4913,  0.5817,  ...,  1.3106,  0.5254,  0.1729]]],       device='cuda:0', grad_fn=<NativeLayerNormBackward0>)), attentions=None)
        # print(logits.shape) # （batchsize,关系类型个数：12）
        output_embedding = result.hidden_states[-1]  # 模型的输出 logits 转换为特定标签的预测分数。具体来说，它将 [batch_size, seq_len, vocab_size] 形状的 logits 转换为 [batch_size, num_labels] 形状的输出
        logits = self.pvp(logits, input_ids)  # 
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.get_loss(logits, input_ids, labels)

        ke_loss = self.ke_loss(output_embedding, labels, so, input_ids)
        loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss  # 分类损失：CEloss + 结构损失：KEloss
        self.log("Train/loss", loss)
        self.log("Train/ke_loss", loss)
        return loss
    
    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        print('--------------------------------------------')
        print(batch) # 长度为5
        print(len(batch))
        print('--------------------------------------------')
        input_ids, attention_mask, token_type_ids, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.loss_fn(logits, labels)
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        


    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)  # mask_idx 即 [MASK] token 在每个序列中的索引
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]  # 从 logits 中提取每个序列中 [MASK] 位置的 logits，结果形状为 [batch_size, vocab_size]，也就是每个样本[Mask]token预测的概率分布
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"  # 确保每个序列中只有一个 [MASK] token
        final_output = mask_output[:,self.word2label]  # 从 mask_output 中选择特定词汇的 logits，结果形状为 [batch_size, num_labels]，只去特定词汇的预测概率值
        # self.word2label为对应标签词的词表索引，这样就可以去到预测这些标签词的概率值
        return final_output
        
    def ke_loss(self, logits, labels, so, input_ids):
        subject_embedding = []
        object_embedding = []
        neg_subject_embedding = []
        neg_object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))

            # random select the neg samples
            st_sub = random.randint(1, logits[i].shape[0] - 6)
            span_sub = random.randint(1, 5)
            st_obj = random.randint(1, logits[i].shape[0] - 6)
            span_obj = random.randint(1, 5)
            neg_subject_embedding.append(torch.mean(logits[i, st_sub:st_sub+span_sub], dim=0))
            neg_object_embedding.append(torch.mean(logits[i, st_obj:st_obj+span_obj], dim=0))
            
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        neg_subject_embedding = torch.stack(neg_subject_embedding)
        neg_object_embedding = torch.stack(neg_object_embedding)
        # trick , the relation ids is concated, 


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = logits[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        
        d_1 = torch.norm(subject_embedding + mask_relation_embedding - object_embedding, p=2) / bsz
        d_2 = torch.norm(neg_subject_embedding + real_relation_embedding - neg_object_embedding, p=2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1.*f(self.args.t_gamma - d_1) - f(d_2 - self.args.t_gamma)
        
        return loss

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

class TransformerLitModelTwoSteps(BertLitModel):
    def configure_optimizers(self):
        no_decay_param = ["bais", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.args.lr_2, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }



class DialogueLitModel(BertLitModel):

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = result.logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels) 
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        


    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
        
def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]

class GPTLitModel(BaseLitModel):
    def __init__(self, model, args , data_config):
        super().__init__(model, args)
        # self.num_training_steps = data_config["num_training_steps"]
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.best_f1 = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits

        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        # f1 = compute_f1(logits, labels)["f1"]
        f1 = f1_score(logits, labels)
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argumenT
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = f1_score(logits, labels)
        # f1 = acc(logits, labels)
        self.log("Test/f1", f1)

from models.trie import get_trie
class BartRELitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None):
        super().__init__(model, args)
        self.best_f1 = 0
        self.first = True

        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)

        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        
        self.tokenizer = tokenizer
        self.trie, self.rel2id = get_trie(args, tokenizer=tokenizer)
        
        self.decode = partial(decode, tokenizer=self.tokenizer)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label  = batch.pop("label")
        loss = self.model(**batch).loss
        self.log("Train/loss", loss)
        return loss
        
        

    def validation_step(self, batch, batch_idx):
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"eval_logits": preds.detach().cpu().numpy(), "eval_labels": true.detach().cpu().numpy()}


    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1 and not self.first:
            self.best_f1 = f1
        self.first = False
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
       

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"test_logits": preds.detach().cpu().numpy(), "test_labels": true.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
