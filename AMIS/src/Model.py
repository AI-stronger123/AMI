import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge
from src.Decoupling_matrix_aggregation import construct_adj

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class MHGCN(nn.Module):
    def l2_normalize(self, x):
        
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)
        return x / norm
    def __init__(self, nfeat, nhid, out, dropout, use_l2_norm=True):
        super(MHGCN, self).__init__()
        """
        # Multilayer Graph Convolution with Relation Interaction Enhancement
        """
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        self.dropout = dropout
        
        self.use_l2_norm = use_l2_norm
        """
        Set the trainable weight of adjacency matrix aggregation
        """

       
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(7, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        self.base_relation_weights = self.weight_b
        
 
        self.relation_interaction = torch.nn.Parameter(torch.FloatTensor(3, 3), requires_grad=True)
        torch.nn.init.normal_(self.relation_interaction, mean=0, std=0.1)
        
    
        self.interaction_strength = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        
      
        self.struct_weight = torch.nn.Parameter(torch.ones(7), requires_grad=True)
        torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

        
        #DisMult
        self.num_relations = 3  
        self.distmult_relations = nn.ParameterList([
            nn.Parameter(torch.ones(out) + torch.randn(out) * 0.01)  
            for _ in range(self.num_relations)
        ])

    def get_distmult_score(self, head_emb, tail_emb, relation_idx):
        
        R = self.distmult_relations[relation_idx]
        
        score = torch.sum(head_emb @ R * tail_emb, dim=-1)
        return score

    def forward(self, feature, A, encode, use_relu=True):
        
        bbp_A = adj_matrix_weight_merge(A, self.weight_b)  
        interaction_A = self.compute_relation_interaction_enhancement(A)  
        
        
        final_A = bbp_A + self.interaction_strength * interaction_A

        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        # Output of single-layer GCN
        U1 = self.gc1(feature, final_A)
        # Output of two-layer GCN
        U2 = self.gc2(U1, final_A)

        struct_adj = construct_adj(encode, self.struct_weight)
        print(self.struct_weight)
        U3 = self.gc1(feature, struct_adj)
        U4 = self.gc2(U3, struct_adj)
        
       
        result = ((U1+U2)/2+U4)/2
        branch1 = (U1+U2)/2
        branch2 = U4

        if self.use_l2_norm:
            result = self.l2_normalize(result)
            branch1 = self.l2_normalize(branch1)
            branch2 = self.l2_normalize(branch2)

        return result, branch1, branch2

    def compute_relation_interaction_enhancement(self, A):
     
        from src.Decoupling_matrix_aggregation import coototensor
        
       
        relations = []
        for i in range(3): 
            rel_matrix = coototensor(A[0][i].tocoo()).to_dense()
            relations.append(rel_matrix)
        
        N = relations[0].shape[0]
        device = relations[0].device
        enhancement = torch.zeros(N, N, device=device)
        
       
        for i in range(3):
            for j in range(3):
                if i != j:
                    
                    coeff = self.relation_interaction[i, j]
                    
                   
                    multiplicative_term = coeff * relations[i] * relations[j]
                   
                    condition_mask = (relations[i] > 0).float()
                    conditional_term = coeff * condition_mask * relations[j]
                    
                 
                    combined_term = 0.6 * multiplicative_term + 0.4 * conditional_term
                    enhancement += combined_term
        
    
        return torch.tanh(enhancement)

    def print_interaction_analysis(self):
        
        print("\n=== 关系交互增强分析 ===")
        interaction_matrix = self.relation_interaction.detach().cpu().numpy()
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    value = interaction_matrix[i, j]
                    effect = "促进" if value > 0.05 else "抑制" if value < -0.05 else "微弱"
                    print(f"关系{i}→关系{j}: {value:.4f} ({effect})")
        
        print(f"交互强度: {self.interaction_strength.item():.4f}")
        print("=======================\n")