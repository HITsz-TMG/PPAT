import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers.models.bert import BertModel, BertTokenizer

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, atten_feats=None, nhead=1, attn_drop=0.1):
        super(GATLayer, self).__init__()
        if atten_feats is None: atten_feats = out_feats
        self.in_features = in_feats
        self.out_features = int(out_feats // nhead)
        self.att_features = int(atten_feats // nhead)
        self.nhead = nhead
        assert out_feats % nhead == 0

        self.Wq = nn.Linear(in_feats, self.att_features * nhead)
        self.Wk = nn.Linear(in_feats, self.att_features * nhead)
        self.Wv = nn.Linear(in_feats, self.out_features * nhead)
        self.norm_dim = math.sqrt(self.out_features)

        self.dropout = nn.Dropout(attn_drop)

    def forward(self, graph, idx, x):
        """
        :param graph: (node_num, node_num)
        :param x: (node_num, in_feats)
        :return:
        """
        key_layer = self.transpose_for_scores(self.Wk(x))  # (H,N,F)
        value_layer = self.transpose_for_scores(self.Wv(x))  # (H,N,F)
        query_layer = self.transpose_for_scores(self.Wq(x))  # (H,N,F)

        attention_mask = (graph.edge[idx].T - 1) * 10000  # (N,N)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (H,N,N)
        attention_scores = attention_scores / math.sqrt(self.att_features)  # (H,N,N)
        attention_scores = attention_scores + attention_mask.unsqueeze(0).expand(self.nhead,-1,-1)  # (H,N,N)

        attention_scores = nn.functional.softmax(attention_scores, dim=-1)  # (H,N,N)
        attention_scores = self.dropout(attention_scores)  # (H,N,N)

        context_layer = torch.matmul(attention_scores, value_layer)  # (H,N,F)
        context_layer = context_layer.permute(1, 0, 2).contiguous()  # (N,H,F)
        context_layer = context_layer.view(context_layer.shape[0], -1)  # (N,H*F)

        return context_layer

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0],self.nhead,-1)
        return x.permute(1, 0, 2)


class CON_GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, atten_feats=None, nhead=1, attn_drop=0.1):
        super(CON_GATLayer, self).__init__()
        if atten_feats is None: atten_feats = out_feats
        self.in_features = in_feats
        self.out_features = int(out_feats // nhead)
        self.att_features = int(atten_feats // nhead)
        self.nhead = nhead
        assert out_feats % nhead == 0
        assert self.att_features % 2 == 0

        self.Wq1 = nn.Linear(in_feats, self.att_features * nhead)
        self.Wk1 = nn.Linear(in_feats, self.att_features * nhead)
        self.Wv1 = nn.Linear(in_feats, self.out_features * nhead // 2)
        self.Wq2 = nn.Linear(in_feats, self.att_features * nhead)
        self.Wk2 = nn.Linear(in_feats, self.att_features * nhead)
        self.Wv2 = nn.Linear(in_feats, self.out_features * nhead // 2)
        self.norm_dim = math.sqrt(self.out_features)

        self.dropout = nn.Dropout(attn_drop)

    def forward(self, graph, idx, x):
        """
        :param graph: (node_num, node_num)
        :param x: (node_num, in_feats)
        :return:
        """
        fst_trans_graph = self.add_row(graph.normal_to_conjugate.to(x.device))  # (2, N*N)
        sec_trans_graph = self.add_row(graph.conjugate_to_normal.to(x.device))  # (2, N*N)
        fst_graph = graph.edge[idx].to(x.device)  # (N,N)
        sec_graph = graph.conjugate_g[idx].to(x.device)  # (N,N)

        k1_layer = self.transpose_for_scores(self.Wk1(x))  # (H,N,F)
        q1_layer = self.transpose_for_scores(self.Wq1(x))  # (H,N,F)
        v1_layer = self.transpose_for_scores(self.Wv1(x))  # (H,N,F)
        k2_layer = self.transpose_for_scores(self.Wk2(x))  # (H,N,F)
        q2_layer = self.transpose_for_scores(self.Wq2(x))  # (H,N,F)
        v2_layer = self.transpose_for_scores(self.Wv2(x))  # (H,N,F)

        att1_scores = torch.matmul(q1_layer, k1_layer.transpose(-1, -2))  # (H,N,N)
        att1_scores = att1_scores / math.sqrt(self.att_features)  # (H,N,N)

        att2_scores = torch.matmul(q2_layer, k2_layer.transpose(-1, -2))  # (H,N,N)
        att2_scores = att2_scores / math.sqrt(self.att_features)  # (H,N,N)

        att1_mask = (fst_graph.T - 1) * 10000  # (N,N)
        vec1 = att1_scores + self.trans(sec_trans_graph, att2_scores) + att1_mask  # (H,N,N)
        vec1 = nn.functional.softmax(vec1, dim=-1)  # (H,N,N)
        vec1 = self.dropout(vec1)  # (H,N,N)
        vec1 = torch.matmul(vec1, v1_layer)  # (H,N,F)
        vec1 = vec1.permute(1, 0, 2).contiguous()  # (N,H,F)
        vec1 = vec1.view(vec1.shape[0], -1)  # (N,H*F)

        att2_mask = (sec_graph.T - 1) * 10000  # (N,N)
        vec2 = att2_scores + self.trans(fst_trans_graph, att1_scores) + att2_mask  # (H,N,N)
        vec2 = nn.functional.softmax(vec2, dim=-1)  # (H,N,N)
        vec2 = self.dropout(vec2)  # (H,N,N)
        vec2 = torch.matmul(vec2, v2_layer)  # (H,N,F)
        vec2 = vec2.permute(1, 0, 2).contiguous()  # (N,H,F)
        vec2 = vec2.view(vec2.shape[0], -1)  # (N,H*F)

        return torch.cat((vec1, vec2), dim=1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.nhead, -1)
        return x.permute(1, 0, 2)

    def trans(self, idx, x: torch.Tensor):
        N = x.shape[1]
        return x[:, idx[0], idx[1]].contiguous().view(-1, N, N)

    def add_row(self, x: torch.Tensor):
        N = x.shape[0]
        y = torch.arange(0, N, dtype=torch.long).to(x)
        y = y.unsqueeze(1).expand(-1, N)
        x = x.view(-1)
        y = y.reshape(-1)
        return torch.stack((y, x), dim=0)


class RGCAT(nn.Module):
    def __init__(
            self,
            in_feats,
            hidden_size,
            logger,
            dropout=0.1,
            attn_drop=0.1,
            nhead=8,
            addition_mode='',
            gat_mode='normal',
    ):
        """Sparse version of GAT."""
        super(RGCAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.hidden_size = hidden_size
        self.addition_mode = addition_mode

        self.inter_fc = nn.Linear(hidden_size, in_feats)
        self.intra_fc = nn.Linear(hidden_size, in_feats)
        self.classify = nn.Linear(in_feats, 2)

        if gat_mode == 'normal':
            self.gat = GATLayer(in_feats+len(addition_mode), hidden_size, hidden_size, nhead, attn_drop)
        elif gat_mode == 'conjugate_cat':
            self.inter_gat = CON_GATLayer(in_feats + len(addition_mode), hidden_size, hidden_size, nhead, attn_drop)
            self.intra_gat = CON_GATLayer(in_feats + len(addition_mode), hidden_size, hidden_size, nhead, attn_drop)
        else:
            raise KeyError

    def forward(self, graph, x):
        """
        :param graph: (layer_num, node_num, node_num)
        :param x: (node_num, in_feats)
        :return:
        """
        layer_num = graph.edge_num()

        y = [self.classify(x)]
        for i in range(layer_num):
            if i == 0:
                z = self.add_info(x, graph, y[-1])
                z = self.intra_gat(graph, i, z)
                z = self.intra_fc(z)
                z = self.dropout(z)

                mask = graph.mask[i].to(x.device).view(-1, 1).expand(-1, z.shape[1])
                x = z * mask + x * (1 - mask)

                y.append(self.classify(x))
            else:
                z = self.add_info(x, graph, y[-1])
                z = self.inter_gat(graph, i, z)
                z = self.inter_fc(z)
                z = self.dropout(z)

                mask = graph.mask[i].to(x.device).view(-1,1).expand(-1,z.shape[1])
                x = z * mask + x * (1 - mask)

                y.append(self.classify(x))

        logits = torch.stack(y,dim=0)
        return logits

    def add_info(self, x, graph, pred):
        if 'intra' in self.addition_mode:
            x = torch.cat((x, graph.intra.to(x.device).view(-1, 1)), dim=1)
        if 'pred' in self.addition_mode:
            pred = F.softmax(pred,dim=1)
            x = torch.cat((x, pred[:,1:2]), dim=1)
        return x


class RRGCATbert_model(nn.Module):
    def __init__(self, config, logger):
        super(RRGCATbert_model, self).__init__()
        self.layer_intra_loss_weight = config.layer_intra_loss_weight
        self.layer_inter_loss_weight = config.layer_inter_loss_weight
        self.need_cls = config.need_cls
        self.feature_size = config.encoder_dim

        self.encoder = BertModel.from_pretrained(config.encoder_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.encoder_path)
        special_tokens_dict = {'additional_special_tokens': ['<t>', '</t>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        rgat_input_dim = 4 * config.encoder_dim if self.need_cls else 2 * config.encoder_dim
        if config.event_emb_mul: rgat_input_dim += rgat_input_dim // 2
        self.rgat = RGCAT(in_feats=rgat_input_dim,
                          hidden_size=config.gat_hidden_size,
                          logger=logger,
                          attn_drop=config.attn_drop,
                          dropout=config.dropout,
                          nhead=config.gat_num_heads,
                          addition_mode=config.addition_mode,
                          gat_mode=config.gat_mode
                          )

        self.sm = nn.Softmax(dim=1)
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

        self.inter_label_weight = config.inter_label_weight
        self.intra_label_weight = config.intra_label_weight
        self.focusing = config.focusing
        self.loss_mode = config.loss_mode
        self.event_emb_mul = config.event_emb_mul
        assert config.loss_mode in ['focal', 'normal']

    def forward(self, input_ids, attn_mask, pos_ids, event_mask, graph, distan, causal=None):
        sent_embedding = self.encoder(input_ids, position_ids=pos_ids, attention_mask=attn_mask)[0]  # sent_num, max_len, feature_size

        # 每个事件 event_mask (event_num,sent_num,max_len)
        event_num = event_mask.shape[0]
        event_sent_embedding = sent_embedding.unsqueeze(0).expand(event_num, -1, -1,-1)  # event_num, sent_num, max_len, feature_size
        event_token_num = event_mask.sum(dim=1).sum(dim=1) + 1e-8  # event_num
        event_token_num = event_token_num.unsqueeze(-1).expand(-1, self.feature_size)  # event_num, feature_size
        event_token_mask = event_mask.unsqueeze(dim=-1).expand(-1, -1, -1,self.feature_size)  # event_num, sent_num, max_len, feature_size
        node_embedding = (event_sent_embedding * event_token_mask).sum(dim=1).sum(dim=1) / event_token_num  # event_num, feature_size

        # 每个事件所在的句子embedding
        if self.need_cls:
            cls_embedding = sent_embedding[:, 0, :]  # sent_num, feature_size
            cls_embedding = cls_embedding.unsqueeze(0).expand(event_num, -1, -1)  # event_num, sent_num, feature_size
            event_cls_mask = event_mask.sum(dim=-1) > 0  # event_num, sent_num
            event_cls_num = event_cls_mask.sum(dim=-1) + 1e-8  # event_num
            event_cls_num = event_cls_num.unsqueeze(-1).expand(-1, self.feature_size)  # event_num, feature_size
            event_cls_mask = event_cls_mask.unsqueeze(-1).expand(-1, -1, self.feature_size)  # event_num, sent_num, feature_size
            event_cls = (cls_embedding * event_cls_mask).sum(dim=1) / event_cls_num  # event_num, feature_size
            node_embedding = torch.cat((node_embedding, event_cls), dim=1)  # event_num, 2*feature_size

        pair_embedding = self.node2pair(node_embedding)
        pair_embedding = pair_embedding.view(event_num * event_num, -1)
        logits = self.rgat(graph, pair_embedding)
        logits = logits.view(logits.shape[0], event_num, event_num, 2)  # layer num, event_num, event_num, 2

        inter = (distan != 0).long()
        if causal is not None:
            loss = None
            for i in range(logits.shape[0]):
                valid = (distan > 0).long()
                lay_loss_mask = self.layer_intra_loss_weight[i] * (1 - valid) + self.layer_inter_loss_weight[i] * valid
                if loss is None:
                    loss = self.criterion(logits[i], causal, inter, lay_loss_mask)
                else:
                    loss += self.criterion(logits[i], causal, inter, lay_loss_mask)

            return loss

        else:
            return logits[-1]

    def criterion(self, logits, label, inter, loss_mask):
        assert loss_mask is not None
        assert inter is not None

        N = logits.shape[0]
        if self.loss_mode == 'normal':
            label_loss = self.CE_loss(logits.view(-1, 2), label.view(-1))
        else:
            label_loss = self.focal_loss(logits.view(-1, 2), label.view(-1))

        pred = label.view(-1) == 1
        weight = torch.ones_like(label_loss).to(label_loss)
        weight[pred & (inter.view(-1) == 0)] = self.intra_label_weight
        weight[pred & (inter.view(-1) == 1)] = self.inter_label_weight
        label_loss = label_loss * weight

        label_loss = label_loss.view(N, N)
        loss_mask = loss_mask - torch.diag_embed(torch.diag(loss_mask))
        loss = loss_mask * label_loss

        return loss.sum()

    def focal_loss(self, logits, label):
        logits = self.sm(logits)
        possibility = logits[:, 1] * (label == 1) + logits[:, 0] * (label == 0)
        label_loss = - ((1 - possibility) ** self.focusing) * torch.log(possibility)
        return label_loss

    def predict(self, input_ids, attn_mask, pos_ids, event_mask, graph, distan, causal, logits=None):
        if logits is None:
            logits = self.forward(input_ids, attn_mask, pos_ids, event_mask, graph, distan)
        logits = logits + logits.transpose(0, 1)
        pred = torch.argmax(logits, dim=2) == 1
        golden = causal == 1

        logits_mask = torch.ones_like(pred, dtype=torch.bool).to(pred)
        logits_mask = logits_mask ^ torch.diag_embed(torch.diag(logits_mask))

        pred = pred & logits_mask
        golden = golden & logits_mask
        right = (pred & golden)

        return right, pred, golden

    def metric(self, input_ids, attn_mask, pos_ids, event_mask, graph, distan, causal, logits=None, analysis=False):
        right, pred, golden = self.predict(input_ids, attn_mask, pos_ids, event_mask, graph, distan, causal, logits=logits)
        inter = (distan != 0).long()
        intra = (distan == 0).long()

        right_num = right.sum()
        pred_num = pred.sum()
        golden_num = golden.sum()

        inter_right_num = (inter & right).sum()
        inter_pred_num = (inter & pred).sum()
        inter_golden_num = (inter & golden).sum()

        intra_right_num = (intra & right).sum()
        intra_pred_num = (intra & pred).sum()
        intra_golden_num = (intra & golden).sum()

        d = {'right num': right_num.item(),
             'pred num': pred_num.item(),
             'golden num': golden_num.item(),

             'inter right num': inter_right_num.item(),
             'inter pred num': inter_pred_num.item(),
             'inter golden num': inter_golden_num.item(),

             'intra right num': intra_right_num.item(),
             'intra pred num': intra_pred_num.item(),
             'intra golden num': intra_golden_num.item(),
            }

        if analysis:
            d['right'] = right
            d['pred'] = pred
            d['golden'] = golden

        return d

    # event embedding to event pair embedding
    def node2pair(self, node_embedding):
        event_num = node_embedding.shape[0]
        node_embedding = node_embedding.unsqueeze(1).expand(-1, event_num, -1)  # event_num,event_num,feature_size
        pair_embedding = torch.cat((node_embedding,node_embedding.transpose(0, 1)),dim=2)  # event_num,event_num,feature_size
        if self.event_emb_mul:
            pair_embedding = torch.cat((pair_embedding,node_embedding * node_embedding.transpose(0, 1)),dim=2)
        return pair_embedding
