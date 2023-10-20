import json
import math
import random
import os
import time
from RRGCATbert_option import option
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from RRGCATbert_graph_build import graph_builder

stack_spliter = None


class RRGCATbert_dataset(Dataset):
    def __init__(self, config, path, tokenizer, logger, mode):
        self.tokenizer = tokenizer
        self.mode = mode
        self.logger = logger
        self.device = config.device
        self.maxlen = config.maxlen
        self.tokenlen = config.tokenlen
        self.shiftlen = config.shiftlen
        self.max_graph_num = config.max_graph
        self.graph_mode = config.graph_mode
        self.conjugate = config.conjugate

        self.graph_builder = graph_builder(default_method=config.graph_mode)

        self.data = []
        self.maxx = 0

        with open(path, 'r') as f:
            self.docs = json.load(f)

        self.CLS = tokenizer.tokenize('[CLS]')
        self.PAD = tokenizer.tokenize('[PAD]')
        self.TSTART = tokenizer.tokenize('<t>')
        self.TEND = tokenizer.tokenize('</t>')
        assert len(self.PAD) == len(self.CLS) == len(self.TSTART) == len(self.TEND) == 1

        if config.data_ckpt != '':
            try:
                self._load(config.data_ckpt + f"{self.mode}_data.pt")
            except:
                self.getdata()
        else:
            self.getdata()

        # self._dump(config.save_path + f"{self.mode}_data.pt", config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input_ids, graph, distan, causal, pos_ids, att_mask, event_mask, docid = self.data[item]

        input_ids = input_ids.to(self.device)
        att_mask = att_mask.to(self.device)
        event_mask = event_mask.to(self.device)
        pos_ids = pos_ids.to(self.device)
        distan = distan.to(self.device)
        causal = causal.to(self.device)

        if 'test' in self.mode:
            return input_ids, att_mask, pos_ids, event_mask, graph, distan, causal, docid

        return input_ids, att_mask, pos_ids, event_mask, graph, distan, causal

    def getdata(self):
        docid = -1
        for doc in tqdm(self.docs):
            docid += 1

            # events
            events = doc['events']
            tid2mid = {}
            for x, tid in enumerate(events):
                str_tid = str(tid)
                assert str_tid not in tid2mid.keys()
                tid2mid[str_tid] = x

            # causal
            causal = torch.zeros((len(events), len(events)), dtype=torch.long)
            for r in doc['causal']:
                e1 = tid2mid[str(r[0])]
                e2 = tid2mid[str(r[1])]
                causal[e1][e2] = causal[e2][e1] = 1

            # distan
            token_sentence = doc['sentence']
            distan = torch.zeros((len(events), len(events)), dtype=torch.long)
            for e1 in range(len(events)):
                for e2 in range(len(events)):
                    distan[e1][e2] = distan[e2][e1] = abs(token_sentence[events[e1][0]] - token_sentence[events[e2][0]])

            # get sentence tokenized
            tokens, events, sentence_start, sentence_end = self.sentence_tokenize(doc['tokens'], events, token_sentence)
            piece_contain_events, sentence_piece = self.sentence_cut_piece(sentence_start, sentence_end, events)
            token_piece, pos_piece, event_mask, atte_mask = self.get_mask(tokens,sentence_piece,piece_contain_events,events)

            graph = self.graph_builder.build_graph((distan==0).long(), self.max_graph_num, conjugate=self.conjugate)
            graph.add((distan==0).long(), 'intra')

            token_ids = []
            for tt in token_piece:
                token_ids.append(torch.tensor(self.tokenizer.convert_tokens_to_ids(tt), dtype=torch.long))
            token_ids = torch.stack(token_ids, dim=0)

            pos_piece = torch.tensor(pos_piece, dtype=torch.long)

            self.data.append([token_ids,graph,distan,causal,pos_piece,atte_mask,event_mask,docid])

    def sentence_tokenize(self, raw_tokens, raw_event, raw_sentence_id):
        raw_token_idx = list(range(len(raw_tokens)))
        tokens = []
        token_idx = []
        sentence_id = []
        for x, t in enumerate(raw_tokens):
            tt = self.tokenizer.tokenize(t)
            tokens += tt
            token_idx += [raw_token_idx[x]] * len(tt)
            sentence_id += [raw_sentence_id[x]] * len(tt)

        idx_map = []
        for i in range(len(token_idx)):
            if i == 0 or token_idx[i] != token_idx[i-1]:
                idx_map.append([])
            idx_map[-1].append(i)

        events = []
        for tid in raw_event:
            ls = []
            for t in tid:
                ls += idx_map[t]
            events.append(ls)

        sentence_start = [0]
        for x in range(1, len(sentence_id)):
            if sentence_id[x] == sentence_id[x - 1]:
                sentence_start.append(sentence_start[-1])
            else:
                sentence_start.append(x)

        sentence_end = [len(sentence_id) - 1]
        for x in range(len(sentence_id) - 2, -1, -1):
            if sentence_id[x] == sentence_id[x + 1]:
                sentence_end.append(sentence_end[-1])
            else:
                sentence_end.append(x)
        sentence_end.reverse()

        return tokens, events, sentence_start, sentence_end

    def sentence_cut_piece(self, sentence_start, sentence_end, events):
        piece_contain_events = []
        sentence_piece = []
        flag = [False] * len(events)

        if self.shiftlen == -1:
            piece_contain_events.append(list(range(len(events))))
            sentence_piece.append([0, len(sentence_end) - 1])
            return piece_contain_events, sentence_piece

        for s in range(0, len(sentence_start), self.shiftlen):
            e = min(s + self.tokenlen - 1,len(sentence_start) - 1)

            els = []
            for j in range(len(events)):
                es = min(events[j])
                ee = max(events[j])
                if s <= es <= ee <= e:
                    els.append(j)
                    flag[j] = True
            if len(els) > 0:
                sentence_piece.append([s, e])
                piece_contain_events.append(els)

            if e == len(sentence_start) - 1: break

        for i in flag:
            assert i

        return piece_contain_events, sentence_piece

    def get_mask(self,tokens,sentence_piece,piece_contain_events,events):
        def tid2pid(x):
            return x+1
        event_mask = torch.zeros((len(events),len(sentence_piece),self.maxlen), dtype=torch.long)
        atten_mask = torch.zeros((len(sentence_piece), self.maxlen, self.maxlen), dtype=torch.long)
        token_pieces = []
        posid_pieces = []
        for i in range(len(sentence_piece)):
            ps, pe = sentence_piece[i]
            token_pie = self.CLS + tokens[ps:pe + 1]
            l = len(token_pie)
            pos_pie = list(range(l))
            atten_mask[i,:l,:l] = 1
            for j in piece_contain_events[i]:
                atten_mask[i, l:l + 2, torch.tensor(events[j], dtype=torch.long) - ps + 1] = 1
                atten_mask[i,l:l+2,l:l+2] = 1
                token_pie += self.TSTART + self.TEND
                pos_pie += [events[j][0] - ps + 1, events[j][-1] - ps + 1]
                assert min(events[j]) == events[j][0] and max(events[j]) == events[j][-1]
                event_mask[j,i,l:l+2] = 1
                l+=2

            if len(token_pie) > self.maxx:
                self.maxx = len(token_pie)
                self.logger.info(f"[max] {self.maxx}")

            pad_len = self.maxlen - len(token_pie)
            assert pad_len >= 0 and len(token_pie) == len(pos_pie)
            token_pie += self.PAD * pad_len
            pos_pie += [0] * pad_len

            token_pieces.append(token_pie)
            posid_pieces.append(pos_pie)

        return token_pieces, posid_pieces, event_mask, atten_mask

    @staticmethod
    def collate_fn(data):
        return data[0]

    def _load(self,path):
        if not os.path.exists(path):
            print(f'no data checkpoint at {path} !!!')
        data = torch.load(path, map_location=torch.device('cpu'))
        config = data['config']
        self.logger.info(f'!!! load dataset from {path} !!!\n{config}')
        self.data = data['data']

    def _dump(self,path,config):
        data = {}
        data['config'] = '\n'.join([(format(arg, '<20') + format(str(config[arg]), '<')) for arg in config.keys()])
        data['data'] = self.data
        torch.save(data,path)


def get_dataloader(config, path, tokenizer, logger, mode):
    return DataLoader(dataset=RRGCATbert_dataset(config, path, tokenizer, logger, mode),
                      batch_size=1,
                      shuffle=('train' in mode),
                      collate_fn=RRGCATbert_dataset.collate_fn)


"""
for tid in doc['events']:
    print(' '.join(map(lambda x:doc['tokens'][x],tid)))
    
for i in range(len(event_mask)):
    for j in range(len(event_mask[i])):
        s = e = -1
        assert sum(event_mask[i][j]) == 0 or sum(event_mask[i][j]) == 2
        for k in range(len(event_mask[i][j])):
            if event_mask[i][j][k]==1:
                if s == -1:
                    s = k
                else:
                    e = k
        if sum(event_mask[i][j]) == 2:
            assert token_piece[j][s] == '<t>' and token_piece[j][e] == '</t>'
            print(' '.join(token_piece[j][pos_piece[j][s]:pos_piece[j][e]+1]),end=' ')
        print('---',end='')
    print('')
   
   
ss = set()
for i in range(len(atte_mask)):
    s = -1
    e = -1
    for j in range(len(atte_mask[i])):
        if token_piece[i][j] == '<t>':
            if s == -1: s = j
        elif token_piece[i][j] == '</t>':
            e = j
    assert s != -1 and e != -1
    for j in range(len(atte_mask[i])):
        if j < s:
            assert sum(atte_mask[i][j]) == s and sum(atte_mask[i][j][:s]) == s
        elif j <= e:
            assert sum(atte_mask[i][j][s:]) == 2
            ls = []
            for k in range(len(atte_mask[i][j][:s])):
                if atte_mask[i][j][k] == 1:
                    ls.append(token_piece[i][k])
            ss.add(' '.join(ls))
        else:
            assert sum(atte_mask[i][j]) == 0
            
         
"""