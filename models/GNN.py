import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import deep_robust_utils as utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import numpy as np

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, APPNP,GraphConv,GCNConv, GATConv, SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, add_self_loops, degree, softmax
import numpy as np


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        
        try:
            import torch_sparse
            if isinstance(adj, torch_sparse.SparseTensor):
                output = torch_sparse.matmul(adj, support)
            else:
                output =adj@support
        except ImportError:
            output = adj@support
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            
            self.layers.append(GraphConvolution(nhid, nclass))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None
    
    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()  
        
        if self.with_bn and hasattr(self, 'bns'):
            for bn in self.bns:
                bn.reset_parameters()
        
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x 
    
    def forward_sampler(self, x, adjs):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x 
    def predict(self, x, adj):
        self.eval()
        return self.forward(x,adj)

    def fit_with_val(self, features, adj, labels, data, syn=True,   
                     train_iters=200, initialize=True, verbose=False, normalize=True):

        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        adj_norm = utils.normalize_adj_tensor(adj,sparse=True)
        self.adj_norm = adj_norm
        self.features = features
        self.loss = F.cross_entropy
        self.labels = labels
        self.syn = syn 

        output,best_acc_val, best_acc_test = self._train_with_val(labels, data, train_iters, verbose, syn)
        return output, best_acc_val, best_acc_test

    def _train_with_val(self, labels, data, train_iters, verbose, syn):

        feat_full, adj_full = data.feats_full.to(self.device), data.adj_full.to(self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        if verbose:
            print('=== training gcn model ===')
        
        best_acc_val = 0
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)
        labels_test = torch.LongTensor(data.labels_test).to(self.device) 
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if syn:
                loss_train = self.loss(output, labels)
            else:
                loss_train = self.loss(output[data.train_idx.to(self.device)], labels[data.train_idx.to(self.device)]) 
                
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)
                loss_val = F.cross_entropy(output[data.val_idx], labels_val)
                acc_val = utils.accuracy(output[data.val_idx], labels_val)
                acc_test = utils.accuracy(output[data.test_idx], labels_test) 

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test
                    self.output = output
                    best_output = output.clone().detach()
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        return best_output, best_acc_val.item(), best_acc_test.item()


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None):
        super(MLP, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass  

        self.layers = nn.ModuleList([])  

        if nlayers == 1:
            self.layers.append(nn.Linear(nfeat, nclass, bias=with_bias))
        else:
            if with_bn:
                self.bns = nn.ModuleList()  
                self.bns.append(nn.BatchNorm1d(nhid))  
            self.layers.append(nn.Linear(nfeat, nhid, bias=with_bias))
            for i in range(nlayers - 2):
                self.layers.append(nn.Linear(nhid, nhid, bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))  
            self.layers.append(nn.Linear(nhid, nclass, bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay if with_relu else 0 
        self.with_relu = with_relu  
        self.with_bn = with_bn  
        self.with_bias = with_bias 
        self.output = None  
        self.best_model = None 
        self.best_output = None  
    
    def reset(self):
        for layer in self.layers:
            layer.reset_parameters() 
        if self.with_bn and hasattr(self, 'bns'):
            for bn in self.bns:
                bn.reset_parameters()
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ix](x)  
                if self.with_relu:
                    x = F.relu(x)  
                x = F.dropout(x, self.dropout, training=self.training)

        self.output = x  
        return x
    
    def forward_sampler(self, x, adjs):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x 

    def fit_with_val(self, features, adj, labels, data, syn=True,   
        train_iters=200, initialize=True, verbose=False, normalize=True):

        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)

        adj_norm = adj
        self.adj_norm = adj_norm
        self.features = features
        self.loss = F.cross_entropy
        self.labels = labels
        self.syn = syn 

        output,best_acc_val, best_acc_test = self._train_with_val(labels, data, train_iters, verbose, syn)
        return output, best_acc_val, best_acc_test
    
    def _train_with_val(self, labels, data, train_iters, verbose, syn):

        feat_full, adj_full = data.feats_full.to(self.device), data.adj_full.to(self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)


        if verbose:
            print('=== training gcn model ===')
        
        best_acc_val = 0

        val_acc_list = []
        test_acc_list = []
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_acc_val = 0
        best_acc_test = 0
        labels_val = torch.LongTensor(data.labels_val).to(self.device)
        labels_test = torch.LongTensor(data.labels_test).to(self.device) 
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if syn:
                loss_train = self.loss(output, labels)
            else:
                loss_train = self.loss(output[data.train_idx.to(self.device)], labels[data.train_idx.to(self.device)]) 
                
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)
                loss_val = F.cross_entropy(output[data.val_idx], labels_val)
                acc_val = utils.accuracy(output[data.val_idx], labels_val)
                acc_test = utils.accuracy(output[data.test_idx], labels_test) 

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test
                    self.output = output.clone().detach()
                    best_out = output.clone().detach()
                    weights = deepcopy(self.state_dict())
                    
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        acc_test = utils.accuracy(self.output[data.test_idx],labels_test)

        return best_out, best_acc_val.item(), best_acc_test.item()


    def predict(self, x, adj):
        self.eval()
        return self.forward(x,adj)


class SelfTrainingGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, alpha=0.5, k=10, mode='gcnmlp'):
        super(SelfTrainingGNN, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.k = k
        self.mode = mode 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gcn = GCN(input_dim, hidden_dim, num_classes, device='cuda')
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        ).cuda()
        
        self.attention = nn.Sequential(
            nn.Linear(2 * num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 2),
        ).cuda()

        self.attention[-1].bias.data = torch.tensor([100.0, 0.0], device='cuda')
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_pretrain_gnn = {'logits': None, 'val_acc': -1.0}
        self.best_pretrain_mlp = {'logits': None, 'val_acc': -1.0}
        self.best_fused = {'logits': None, 'val_acc': -1.0}
        
        self.global_best_val_acc = -1.0
        self.global_best_test_acc = -1.0
    
    def reset_encoders(self):
        self.gcn = GCN(self.input_dim, self.hidden_dim, self.num_classes, device='cuda').cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        ).cuda()
        
        self.attention = nn.Sequential(
            nn.Linear(2 * self.num_classes, self.num_classes),
            nn.ReLU(),
            nn.Linear(self.num_classes, 2),
        ).cuda()

        self.attention[-1].bias.data = torch.tensor([100.0, 0.0], device='cuda')

    def train_single_encoder(self, encoder, encoder_type, x, adj, train_mask, labels, val_mask, test_mask, labels_real, epochs=50, lr=0.01):
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=5e-4)
        best_val_acc = -1.0
        best_logits = None
        encoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            if encoder_type in ['gcn']:
                logits = encoder(x, adj) if encoder_type == 'gcn' else encoder(x, adj, use_cached=True)
            else:
                logits = encoder(x)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            
            encoder.eval()
            with torch.no_grad():
                if encoder_type in ['gcn']:
                    logits_val = encoder(x, adj) if encoder_type == 'gcn' else encoder(x, adj, use_cached=True)
                else:
                    logits_val = encoder(x)
                pred = F.softmax(logits_val, dim=1).argmax(dim=1)
                val_acc = (pred[val_mask] == labels_real[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_logits = logits_val.clone()
            encoder.train()
        return encoder, best_val_acc, best_logits

    def forward(self, x, adj):
        gnn_logits = self.gcn(x, adj)
        mlp_logits = self.mlp(x)
        concat_logits = torch.cat([gnn_logits, mlp_logits], dim=1)
        attn_scores = self.attention(concat_logits) 
        attn_weights = F.softmax(attn_scores, dim=1)  
        fused_logits = attn_weights[:, 0:1] * gnn_logits + attn_weights[:, 1:2] * mlp_logits 
        return gnn_logits, mlp_logits, fused_logits

    def train_fused(self, x, adj, gnn_train_mask, mlp_train_mask, fused_train_mask, labels, val_mask, test_mask, labels_real, epochs=50, lr=0.01,
                    w_gnn=1.0, w_mlp=1.0, w_fused = 1.0
                    ):
        params = list(self.gcn.parameters())+list(self.mlp.parameters())+list(self.attention.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
        
        best_val_acc = -1.0
        best_logits = {'gnn': None, 'mlp': None, 'fused': None}
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            gnn_logits, mlp_logits, fused_logits = self.forward(x, adj)
            loss_gnn = self.criterion(gnn_logits[gnn_train_mask], labels[gnn_train_mask])
            loss_mlp = self.criterion(mlp_logits[mlp_train_mask], labels[mlp_train_mask])
            loss_fused = self.criterion(fused_logits[fused_train_mask], labels[fused_train_mask])
            total_loss = w_gnn * loss_gnn + w_fused*loss_fused + w_mlp * loss_mlp 
            total_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                _, _, fused_logits_val = self.forward(x, adj)
                fused_pred = F.softmax(fused_logits_val, dim=1).argmax(dim=1)
                val_acc = (fused_pred[val_mask] == labels_real[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_logits['gnn'] = gnn_logits.clone()
                best_logits['mlp'] = mlp_logits.clone()
                best_logits['fused'] = fused_logits_val.clone()
    
        return best_logits, best_val_acc

    def generate_pseudo_labels(self, gnn_probs, mlp_probs, current_unlabeled_mask, ratio):
        unlabeled_indices = torch.where(current_unlabeled_mask)[0]
        num_unlabeled = len(unlabeled_indices)
        if num_unlabeled == 0:
            return (torch.tensor([], device=gnn_probs.device), torch.tensor([], device=gnn_probs.device),
                    torch.tensor([], device=gnn_probs.device), torch.tensor([], device=gnn_probs.device),
                    torch.tensor([], device=gnn_probs.device), torch.tensor([], device=gnn_probs.device))
        
        num_select_per_model = max(1, int(num_unlabeled * ratio))
        num_select_per_model = min(num_select_per_model, num_unlabeled)
        
        gnn_conf = gnn_probs[unlabeled_indices].max(dim=1).values
        _, gnn_top_indices = torch.topk(gnn_conf, num_select_per_model)
        gnn_selected = unlabeled_indices[gnn_top_indices]
        selected_indices = gnn_selected

        gnn_max_conf, gnn_pred = gnn_probs[selected_indices].max(dim=1)
        mlp_max_conf, mlp_pred = mlp_probs[selected_indices].max(dim=1)
        
        consistent_mask = (gnn_pred == mlp_pred)
        consistent_indices = selected_indices[consistent_mask]
        consistent_labels = gnn_pred[consistent_mask]
        common_indices = consistent_indices
        common_labels = consistent_labels
        
        inconsistent_mask = ~consistent_mask
        inconsistent_indices = selected_indices[inconsistent_mask]
        inconsistent_gnn_conf = gnn_max_conf[inconsistent_mask]
        inconsistent_mlp_conf = mlp_max_conf[inconsistent_mask]
        inconsistent_gnn_pred = gnn_pred[inconsistent_mask]
        inconsistent_mlp_pred = mlp_pred[inconsistent_mask]
        gnn_better_mask = inconsistent_gnn_conf >= inconsistent_mlp_conf 

        gnn_pseudo_indices = torch.cat([common_indices, inconsistent_indices[gnn_better_mask]])
        gnn_pseudo_labels = torch.cat([common_labels, inconsistent_gnn_pred[gnn_better_mask]])
        
        mlp_better_mask = inconsistent_gnn_conf < inconsistent_mlp_conf 
        mlp_pseudo_indices = torch.cat([common_indices, inconsistent_indices[mlp_better_mask]])
        mlp_pseudo_labels = torch.cat([common_labels, inconsistent_mlp_pred[mlp_better_mask]])
        
        fused_pseudo_indices = torch.unique(torch.cat([gnn_pseudo_indices, mlp_pseudo_indices]))
        fused_pseudo_labels = []
        for idx in fused_pseudo_indices:
            if idx in gnn_pseudo_indices:
                label = gnn_pseudo_labels[gnn_pseudo_indices == idx].item()
            else:
                label = mlp_pseudo_labels[mlp_pseudo_indices == idx].item()
            fused_pseudo_labels.append(label)
        fused_pseudo_labels = torch.tensor(fused_pseudo_labels, device=gnn_probs.device, dtype=torch.long)

        return gnn_pseudo_indices, gnn_pseudo_labels, mlp_pseudo_indices, mlp_pseudo_labels, fused_pseudo_indices, fused_pseudo_labels
    
    def self_train(self, x, adj, labels, train_mask, val_mask, test_mask, unlabeled_mask,
                   num_stages=10, pretrain_stages=0,  
                   initial_ratio=0.1, ratio_step=0.10, epochs_per_stage=600, w_gnn= 1.0, w_mlp=1.0, w_fused=1.0):
        labels_real = labels.detach().clone()
        N = x.size(0)
        
        current_gnn_train_mask = train_mask.clone()
        current_mlp_train_mask = train_mask.clone()
        current_fused_train_mask = train_mask.clone()  
        current_unlabeled_mask = unlabeled_mask.clone()
        current_adj = adj.clone()
        current_ratio = initial_ratio
        
        print(f"=== Start two phase self-training (stages): {pretrain_stages}, total stages: {num_stages}) ===")
        print(f"Init labeled samples: {train_mask.sum().item()}, init unlabled samples: {current_unlabeled_mask.sum().item()}\n")

        gnn_encoder = self.gcn
        mlp_encoder =self.mlp 
        print("start pretraining")
        for stage in range(pretrain_stages):
            gnn_encoder = self.gcn if self.mode == 'gcnmlp' else self.ppr_proj
            encoder_type = 'gcn' if self.mode == 'gcnmlp' else 'ppr'
            gnn_encoder, gnn_val_acc, gnn_logits = self.train_single_encoder(
                gnn_encoder, encoder_type, x, current_adj, current_gnn_train_mask, labels,
                val_mask, test_mask, labels_real, epochs=epochs_per_stage
            )
            mlp_encoder, mlp_val_acc, mlp_logits = self.train_single_encoder(
                self.mlp, 'mlp', x, current_adj, current_mlp_train_mask, labels,
                val_mask, test_mask, labels_real, epochs=epochs_per_stage
            )

            if gnn_val_acc > self.best_pretrain_gnn['val_acc']:
                self.best_pretrain_gnn = {'logits': gnn_logits.clone(), 'val_acc': gnn_val_acc}
            if mlp_val_acc > self.best_pretrain_mlp['val_acc']:
                self.best_pretrain_mlp = {'logits': mlp_logits.clone(), 'val_acc': mlp_val_acc}
            print(f"Pretrain stage {stage+1}/{pretrain_stages} - GNN Val Acc: {gnn_val_acc:.4f}, MLP Val Acc: {mlp_val_acc:.4f}")
            
            gnn_probs = F.softmax(gnn_logits, dim=1)
            mlp_probs = F.softmax(mlp_logits, dim=1)
            gnn_pseudo_indices, gnn_pseudo_labels, mlp_pseudo_indices, mlp_pseudo_labels, fused_pseudo_indices, fused_pseudo_labels = self.generate_pseudo_labels(
                gnn_probs, mlp_probs, current_unlabeled_mask, ratio=current_ratio
            )
            
            if len(gnn_pseudo_indices) > 0:
                current_gnn_train_mask[gnn_pseudo_indices] = True
                labels[gnn_pseudo_indices] = gnn_pseudo_labels
            if len(mlp_pseudo_indices) > 0:
                current_mlp_train_mask[mlp_pseudo_indices] = True
                labels[mlp_pseudo_indices] = mlp_pseudo_labels
            pseudo_all_indices = fused_pseudo_indices
            current_unlabeled_mask[pseudo_all_indices] = False
            current_ratio = min(current_ratio + ratio_step, 1.0)
            
            print(f"Pretrain stages {stage+1} - number of unlabeled samples: {current_unlabeled_mask.sum().item()}")
        
        print("\nBest pretrained GNN Val Acc: {:.4f}, MLP Val Acc: {:.4f}".format(
            self.best_pretrain_gnn['val_acc'], self.best_pretrain_mlp['val_acc']
        ))
        print("="*50 + "\n")

        print("Start fusion training")
        self.gcn.load_state_dict(gnn_encoder.state_dict())  
        self.mlp.load_state_dict(mlp_encoder.state_dict())  

        fusion_stages = max(num_stages-pretrain_stages, 1)
        for stage in range(fusion_stages):
            print(f"\nFusion stage {stage+1}/{fusion_stages}")

            best_logits, val_acc = self.train_fused(
                x, current_adj, 
                gnn_train_mask=current_gnn_train_mask, 
                mlp_train_mask=current_mlp_train_mask, 
                fused_train_mask=current_fused_train_mask,
                labels=labels,   
                val_mask=val_mask, 
                test_mask=test_mask, 
                labels_real=labels_real, 
                epochs= epochs_per_stage, 
                w_gnn=w_gnn, 
                w_mlp=w_mlp, 
                w_fused= w_fused 
            )
            
            if val_acc > self.best_fused['val_acc']:
                self.best_fused = {'logits': best_logits['fused'].clone(), 'val_acc': val_acc}
            print(f"Fusion stage {stage+1} - Val Acc: {val_acc:.4f}")
            
            gnn_probs = F.softmax(best_logits['gnn'], dim=1)
            mlp_probs = F.softmax(best_logits['mlp'], dim=1)
            fuse_probs = F.softmax(best_logits['fused'],dim=1)

            gnn_pseudo_indices, gnn_pseudo_labels, mlp_pseudo_indices, mlp_pseudo_labels, fused_pseudo_indices, fused_pseudo_labels = self.generate_pseudo_labels(
                gnn_probs, mlp_probs, current_unlabeled_mask, ratio=current_ratio
            )

            if len(gnn_pseudo_indices) > 0:
                current_gnn_train_mask[gnn_pseudo_indices] = True
                labels[gnn_pseudo_indices] = gnn_pseudo_labels
            if len(mlp_pseudo_indices) > 0:
                current_mlp_train_mask[mlp_pseudo_indices] = True
                labels[mlp_pseudo_indices] = mlp_pseudo_labels
            if len(fused_pseudo_indices) > 0:
                current_fused_train_mask[fused_pseudo_indices] = True
                labels[fused_pseudo_indices] = fused_pseudo_labels
            pseudo_all_indices = fused_pseudo_indices
            current_unlabeled_mask[pseudo_all_indices] = False
            current_ratio = min(current_ratio + ratio_step, 1.0)
            
            print(f"Fusion stage {stage+1} - number of unlabeled samples: {current_unlabeled_mask.sum().item()}")
            if current_unlabeled_mask.sum().item() == 0:
                print("No unlabeled, early stop")
                break
        
        print("\n" + "="*50)
        print("Finish the fusion")
        print(f"Best fusion model Val Acc: {self.best_fused['val_acc']:.4f}")
        final_logits = self.best_fused['logits']
        final_pred = F.softmax(final_logits, dim=1).argmax(dim=1)
        final_val_acc = (final_pred[val_mask] == labels_real[val_mask]).float().mean().item()
        final_test_acc = (final_pred[test_mask] == labels_real[test_mask]).float().mean().item()
        print(f"Final test Acc: {final_test_acc:.4f}")
        return final_logits, current_adj, final_pred, final_val_acc, final_test_acc
