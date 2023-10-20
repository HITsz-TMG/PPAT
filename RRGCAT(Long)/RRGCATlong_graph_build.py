import json
import numpy as np
import torch
from attrdict import AttrDict


class pair_graph_template:
    def __init__(self, event_num):
        self.event_num = event_num

        self.ls = []
        pid = 0
        for i in range(event_num):
            for j in range(event_num):
                self.ls.append((pid, i, j))
                assert self.nid2pid(i, j) == pid
                assert self.pid2nid(pid) == (i, j)
                pid += 1
        self.pair_num = len(self.ls)

        # build full graph
        self.full_graph = self.init_graph()
        self.neighbor = []
        for master_pid in range(self.pair_num):
            ns = [master_pid]
            a, b = self.pid2nid(master_pid)
            if a != b:
                for c in range(event_num):
                    if c == a or c == b: continue
                    if a < b:
                        neighbor_pid = self.nid2pid(a, c) if a < c else self.nid2pid(c, a)
                    else:
                        neighbor_pid = self.nid2pid(a, c) if a > c else self.nid2pid(c, a)

                    self.full_graph[master_pid][neighbor_pid] = 1
                    ns.append(neighbor_pid)

                    if a < b:
                        neighbor_pid = self.nid2pid(c, b) if c < b else self.nid2pid(b, c)
                    else:
                        neighbor_pid = self.nid2pid(c, b) if c > b else self.nid2pid(b, c)

                    self.full_graph[master_pid][neighbor_pid] = 1
                    ns.append(neighbor_pid)
            self.neighbor.append(ns)

        self.con_neighbor = None

        self.normal_to_conjugate = torch.stack([torch.arange(0, self.pair_num) for i in range(self.pair_num)], dim=0)
        self.conjugate_to_normal = torch.stack([torch.arange(0, self.pair_num) for i in range(self.pair_num)], dim=0)
        for i in range(self.event_num):
            for j in range(self.event_num):
                master_pid = self.nid2pid(i, j)
                nor_neighbor = self.get_neighbor_pid(master_pid)
                con_neighbor = self.get_conneighbor_pid(master_pid)
                for k in range(len(nor_neighbor)):
                    nor_pid = nor_neighbor[k]
                    con_pid = con_neighbor[k]
                    self.normal_to_conjugate[master_pid][con_pid] = nor_pid
                    self.conjugate_to_normal[master_pid][nor_pid] = con_pid

    def nid2pid(self, i, j):
        return i * self.event_num + j

    def pid2nid(self, pid):
        i = pid // self.event_num
        j = pid - self.event_num * i
        return i, j

    def init_graph(self):
        return torch.eye(self.pair_num, dtype=torch.long)

    def node_iter(self):
        return self.ls

    def get_neighbor_pid(self, pid):
        return self.neighbor[pid]

    def get_transpose_pid(self, pid):
        i, j = self.pid2nid(pid)
        return self.nid2pid(j, i)

    def get_conneighbor_pid(self, pid):
        def same_direct(ii, jj, kk, reverse=False):
            if reverse:
                if ii < kk:
                    return (kk, jj) if jj < kk else (jj, kk)
                else:
                    return (kk, jj) if jj > kk else (jj, kk)
            else:
                if ii < kk:
                    return (jj, kk) if jj < kk else (kk, jj)
                else:
                    return (jj, kk) if jj > kk else (kk, jj)

        if self.con_neighbor is None:
            self.con_neighbor = []
            for mpid in range(self.pair_num):
                con_dir = {}
                i, j = self.pid2nid(mpid)
                if i == j:
                    con_dir[(i, j)] = (i, j)
                else:
                    con_dir[(i, j)] = (i, j)
                    con_dir[(j, i)] = (j, i)
                    for k in range(self.event_num):
                        if k == i or k == j: continue
                        con_dir[(i, k)] = same_direct(i, j, k)
                        con_dir[(k, i)] = same_direct(i, j, k, reverse=True)
                        con_dir[(j, k)] = same_direct(j, i, k)
                        con_dir[(k, j)] = same_direct(j, i, k, reverse=True)
                con_nei = []
                for npid in self.neighbor[mpid]:
                    a, b = self.pid2nid(npid)
                    c, d = con_dir[(a, b)]
                    cpid = self.nid2pid(c, d)
                    con_nei.append(cpid)
                self.con_neighbor.append(con_nei)

        return self.con_neighbor[pid]


class pair_graph:
    def __init__(self):
        self.edge = []
        self.mask = []
        self.conjugate_g = []
        self.conjugate_to_normal = None
        self.normal_to_conjugate = None
        self.intra = None

    def add(self, value, name):
        if name == 'edge':
            self.edge.append(value)
        elif name == 'mask':
            self.mask.append(value)
        elif name == 'intra':
            self.intra = torch.tensor(value)
        else:
            assert False

    def to(self, device):
        for i in range(self.edge_num()):
            self.edge[i] = self.edge[i].to(device)
        for i in range(len(self.mask)):
            self.mask[i] = self.mask[i].to(device)
        self.intra = self.intra.to(device)
        if len(self.conjugate_g) != 0:
            for i in range(self.edge_num()):
                self.conjugate_g[i] = self.conjugate_g[i].to(device)
            self.conjugate_to_normal = self.conjugate_to_normal.to(device)
            self.normal_to_conjugate = self.normal_to_conjugate.to(device)

    def edge_num(self):
        return len(self.edge)


class graph_builder:
    def __init__(self,
                 catch=True,
                 default_method='full',
                 ):
        self.catch = None
        if catch: self.catch = {}
        self.default_method = default_method

    def build_graph(self, intra_node, max_graph, method=None, conjugate=False, simplify=False):
        if method is None: method = self.default_method

        if method == 'full':
            graph = self.full(intra_node, max_graph, conjugate)

        elif method == 'pipeline':
            graph = self.pipeline(intra_node, max_graph, conjugate)

        else:
            assert False

        return graph

    def full(self, intra_node, max_graph, conjugate):
        """
        全连接
        """
        graph, graph_template = self.__graph_init(intra_node)

        for layer in range(max_graph):
            g = graph_template.init_graph()

            for i in graph_template.node_iter():
                master_pid = i[0]
                neighbors = graph_template.get_neighbor_pid(master_pid)
                for neighbor_pid in neighbors:
                    g[neighbor_pid][master_pid] = 1

            graph.add(g, 'edge')

            if conjugate:
                self.build_con_graph(graph, graph_template, g, check=True)

        return graph

    def pipeline(self, intra_node, max_graph, conjugate):
            """
            只有句内到句外的边
            """
            graph, graph_template = self.__graph_init(intra_node)

            intra_fla = intra_node.flatten().unsqueeze(-1)
            intra_mask = torch.matmul(intra_fla,intra_fla.T)
            intra_graph = (intra_mask * graph_template.full_graph | torch.eye(intra_mask.shape[0],dtype=torch.long)).long()

            graph.add(intra_graph, 'edge')
            graph.add(intra_node, 'mask')
            inter_node = 1 - intra_node
            for layer in range(max_graph):
                graph.add(graph_template.full_graph, 'edge')
                graph.add(inter_node, 'mask')

            if conjugate:
                graph.normal_to_conjugate = graph_template.normal_to_conjugate
                graph.conjugate_to_normal = graph_template.conjugate_to_normal
                graph.conjugate_g.append(intra_graph)
                for layer in range(max_graph):
                    graph.conjugate_g.append(graph_template.full_graph)

            return graph

    def __graph_init(self, intra_node):
        graph = pair_graph()
        graph_template = self.get_template_graph(intra_node.shape[0])
        return graph, graph_template

    def get_template_graph(self, event_num):
        if self.catch is not None:
            graph_template = self.catch.get(str(event_num))
            if graph_template is None:
                graph_template = pair_graph_template(event_num)
                self.catch[str(event_num)] = graph_template
        else:
            graph_template = pair_graph_template(event_num)
        return graph_template
