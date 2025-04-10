import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rnaglib.rnattentional.layers import RGATLayer
import numpy as np
import os
#import resnet50_1d, resnet50_2d
from Model.se_resnet import *
from Model.capsnet import *


class RGATEmbedder(nn.Module):
    """
    This is an exemple GAT for unsupervised learning, going from one element of "dims" to the other

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 dims,
                 num_heads=3,
                 sample_other=0.2,
                 infeatures_dim=0,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 return_loss=True,
                 verbose=False):
        super(RGATEmbedder, self).__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.use_node_features = (infeatures_dim != 0)
        self.in_dim = 1 if infeatures_dim == 0 else infeatures_dim
        self.conv_output = conv_output
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.verbose = verbose
        self.return_loss = return_loss
        
        self.layers = self.build_model()

        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = self.dims[-2:]
        if self.verbose:
            print("short, ", short)
            print("last_hidden, last ", last_hidden, last)

        # input feature is just node degree
        i2h = RGATLayer(in_feat=self.in_dim,
                        out_feat=self.dims[0],
                        num_rels=self.num_rels,
                        num_bases=self.num_bases,
                        num_heads=self.num_heads,
                        sample_other=self.sample_other,
                        activation=F.relu,
                        self_loop=self.self_loop)
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                            out_feat=dim_out,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            layers.append(h2h)

        # hidden to output
        if self.conv_output:
            h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                            out_feat=last,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            self_loop=self.self_loop,
                            activation=None)
        else:
            h2o = nn.Linear(last_hidden * self.num_heads, last)
        layers.append(h2o)
        return layers

    def deactivate_loss(self):
        for layer in self.layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g, features,mod = 1):
        iso_loss = 0
        if self.use_node_features:
            if mod == 1:
                h = g.ndata['features'].to(self.current_device)
            else:
                h = features
        else:
            # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            if not self.conv_output and (i == len(self.layers) - 1):
                h = layer(h)
            else:
                if layer.return_loss:
                    h, loss = layer(g=g, feat=h)
                    iso_loss += loss
                else:
                    #print(features.shape)
                    h = layer(g=g, feat=h)
                    #print(h.shape)
        if self.return_loss:
            return h, iso_loss
        else:
            return h


class Attention(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        # 定义查询、键、值的线性变换层
        self.q_linear = nn.Linear(self.head_dim, out_features)
        self.k_linear = nn.Linear(self.head_dim, out_features)
        self.v_linear = nn.Linear(self.head_dim, out_features)
        #维度变换
        self.proj_linear = nn.Linear(in_features, self.head_dim * self.num_heads)
    def forward(self, x):
        # 将输入矩阵 x 分割成 num_heads 份
        seq_len = x.shape[0]
        # batch_size = x.shape[0]
        x = x.unsqueeze(0)
        batch_size = 1
        #x = self.proj_linear(x)  # (batch_size, seq_len, self.num_heads * self.head_dim)
        x = x.view(batch_size,seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 对查询、键、值进行线性变换
        q = self.q_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.k_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        v = self.v_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # 对节点特征进行加权求和
        out = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(seq_len, -1)  # (batch_size, seq_len, num_heads * head_dim)
        return out.squeeze(0)


class MLNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.5):
        super(MLNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        # self.dropout3 = nn.Dropout(dropout_prob)
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
        # x = self.dropout3(self.bn3(torch.relu(self.fc3(x))))
        x = self.sigmoid(self.out(x))
        return x

class RGATClassifier(nn.Module):
    """
    This is an exemple GAT for supervised learning, that uses the previous Embedder network

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 rgat_embedder,
                 ernie_embedder = None,
                 rgat_embedder_pre = None,
                 capsnet = None,
                 classif_dims=None,
                 num_heads=5,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 verbose=False,
                 return_loss=True,
                 sample_other=0.2):
        super(RGATClassifier, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.conv_output = conv_output
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.return_loss = return_loss
        if ernie_embedder != None:
            self.ernie_dim = 768
        else:
            self.ernie_dim = 768
        self.feature_dim = 12 #asa 5 #torsion 18 #dbn 12
        self.rgat_embedder = rgat_embedder
        self.rgat_embedder_pre = rgat_embedder_pre
        self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads + self.feature_dim + self.ernie_dim + self.feature_dim + 128 * 1

        if self.rgat_embedder_pre!=None:
            self.last_dim_embedder += rgat_embedder_pre.dims[-1] * rgat_embedder_pre.num_heads
        self.classif_dims = classif_dims
        self.ernie_embedder = ernie_embedder	
        self.classif_layers = self.build_model()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.verbose = verbose
        if self.verbose:
            print(self.classif_layers)
            print("Num rels: ", self.num_rels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        
        self.window = 7
        self.padsize = int(self.window/3)
        self.localpool = nn.MaxPool2d(kernel_size= 5, stride=1)
        self.localpad = nn.ConstantPad2d((self.padsize, self.padsize,self.padsize,self.padsize), value=0)

        #att
        att_input =350
        att_output =128
        att_num_head =8
        att_mlpinput = att_output * att_num_head
        att_mlphidden = 32
        self.att = Attention(att_input, att_output, att_num_head)
        self.mlp = MLNet(att_mlpinput, att_mlphidden)
        self.cutoff_len = 440

        #self.resnet = SENet(BasicBlock, [2, 2, 2, 2])
        #self.capsnet = CapsuleNet()

        kernels = [13, 17, 21]   
        padding1 = (kernels[1]-1)//2
        self.conv2d_1 = torch.nn.Sequential()
        self.conv2d_1.add_module("conv2d_1",torch.nn.Conv2d(1,128,padding= (padding1,0),kernel_size=(kernels[1],self.feature_dim)))
        self.conv2d_1.add_module("relu1",torch.nn.ReLU())
        self.conv2d_1.add_module("pool2",torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1))
        

        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("Dense1", torch.nn.Linear(self.last_dim_embedder,192))
        self.DNN1.add_module("Relu1", torch.nn.ReLU())
        self.dropout_layer = nn.Dropout(0.1)
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("Dense2", torch.nn.Linear(192,96))#192,96
        self.DNN2.add_module("Relu2", torch.nn.ReLU())
        self.dropout_layer2 = nn.Dropout(0.1)
        self.outLayer = nn.Sequential(
            torch.nn.Linear(96, 1), #96
            torch.nn.Sigmoid())

        self.fc2d1 = nn.Linear(128, 64)
        self.fc2d2 = nn.Linear(64, 1)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for kernel_size in [5, 10, 20]:
            padding = (kernel_size - 1) // 2
            conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=padding)
            bn = nn.BatchNorm1d(32)
            pool = nn.MaxPool1d(kernel_size=2)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.pool_layers.append(pool)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv11 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=100)
        self.conv12 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=175)
        self.conv13 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=350)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 172, 128)#84 191  64 84 #172#21
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for kernel_size in [5, 10, 20]:
            padding = (kernel_size - 1) // 2
            conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=padding)
            bn = nn.BatchNorm1d(32)
            pool = nn.MaxPool1d(kernel_size=2)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.pool_layers.append(pool)


    def build_model(self):
        if self.classif_dims is None:
            return self.rgat_embedder

        classif_layers = nn.ModuleList()
        # Just one convolution
        if len(self.classif_dims) == 1:
            if self.conv_output:
                h2o = RGATLayer(in_feat=self.last_dim_embedder,
                                out_feat=self.classif_dims[0],
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                # Old fix for a bug in dgl<0.6
                                # self_loop=self.self_loop and self.classif_dims[0] > 1,
                                activation=None)
            else:
                h2h = nn.Linear(self.last_dim_embedder, self.last_dim_embedder)
                classif_layers.append(h2h)
                h2h2 = nn.Linear(self.last_dim_embedder, self.last_dim_embedder)
                classif_layers.append(h2h2)
                h2o = nn.Linear(self.last_dim_embedder, self.classif_dims[0])
                 
            classif_layers.append(h2o)
            
            return classif_layers

        # The supervised is more than one layer
        else:
            i2h = RGATLayer(in_feat=self.last_dim_embedder,
                            out_feat=self.classif_dims[0],
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            classif_layers.append(i2h)
            last_hidden, last = self.classif_dims[-2:]
            short = self.classif_dims[:-1]
            for dim_in, dim_out in zip(short, short[1:]):
                h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                                out_feat=dim_out,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                activation=F.relu,
                                self_loop=self.self_loop)
                classif_layers.append(h2h)

            # hidden to output
            if self.conv_output:
                h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                                out_feat=last,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                activation=None)
            else:
                h2o = nn.Linear(last_hidden * self.num_heads, last)
            classif_layers.append(h2o)
            return classif_layers

    def deactivate_loss(self):
        self.return_loss = False
        self.rgat_embedder.deactivate_loss()
        for layer in self.classif_layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g , features, indxs, seqs_old, seqs, seqlens, len_graphs, chain_idx, rna, chain_id):
        iso_loss = 0
        g_copy = g.clone().to(self.current_device)
        cnt = 1

        if self.rgat_embedder.return_loss:
            h, loss = self.rgat_embedder(g,features,0)
        else:
            h = self.rgat_embedder(g,features,0)
            loss = 0
        #print(features[0])
        if self.rgat_embedder_pre!=None:
            cnt = 2
            if self.rgat_embedder_pre.return_loss:
                #print(next(self.rgat_embedder_pre.parameters()).device)
            
                h_pre, loss_pre = self.rgat_embedder_pre(g_copy,features,1)
                h = torch.cat((h,h_pre),1)

            else:
                h_pre = self.rgat_embedder_pre(g_copy,features,1)
                h = torch.cat((h,h_pre),1)
                loss = 0
        iso_loss += loss
        #print(features)
        #print(h.shape) #192
        h = torch.cat((h,features),1)
        features = torch.index_select(features, 0, indxs)
        seq_embedding = torch.tensor([]).to(self.current_device).reshape(-1,768)
        h = torch.index_select(h, 0, indxs)
        #print(h.shape) #192+feature_dim
        seqlens_new=[seqlens[x] for x in chain_idx]
        #print(seqlens_new)

        for rna_id, c_id, seq_len, c_idx in zip(rna, chain_id, seqlens_new, chain_idx):               
            print(rna_id)
            ernie_embed_path='/home/MultiModRLBP/data/RLBP_embedding2/'+rna_id[0:4]+'_'+c_id.upper()+'_all_embedding.npy'
            #print(ernie_embed_path)
            seq_encode = np.load(ernie_embed_path)
            seq_encode = torch.tensor(seq_encode).to(torch.float32)
            #print(seq_encode.shape)
            seqlen = seq_encode.shape[1]-2
            
            if seq_len-seqlen == 1:
                seq_embedding = torch.cat((seq_embedding, seq_encode[0,1:seqlen+2,:].to(self.current_device).reshape(-1, 768)), dim=0)
            else:
                seq_embedding = torch.cat((seq_embedding, seq_encode[0,1:seqlen+1,:].to(self.current_device).reshape(-1, 768)), dim=0) 
            #print(seq_embedding.shape)

        pos = 0
        idx = 0
        mlp = 1
        #print(h.shape)
        single_chain = torch.tensor([]).to(self.current_device).reshape(-1,192*cnt+self.feature_dim*mlp)
        single_features = torch.tensor([]).to(self.current_device).reshape(-1,self.feature_dim)
        for seqlen in seqlens:
            if idx in chain_idx:
                #print(seqlen)
                single_chain = torch.cat((single_chain, h[pos:pos+seqlen].to(self.current_device).reshape( -1, 192*cnt+self.feature_dim*mlp)), dim=0)
                single_features = torch.cat((single_features, features[pos:pos+seqlen].to(self.current_device).reshape( -1, self.feature_dim)), dim=0)
            idx += 1
            pos += seqlen        
        #print(single_chain.shape)
        h = torch.cat((single_chain,seq_embedding), 1)

        fixed_features = torch.tensor([]).reshape(-1,self.feature_dim).to(self.current_device)
        pos = 0
        idx = 0

        for idx in chain_idx:
            
            L = seqlens[idx]

            local_h = single_features[pos:pos+L].to(self.current_device)
            #print(local_h.shape)
            for i in range(L,self.cutoff_len):
                add = torch.tensor([[0 for i in range(self.feature_dim)]]).to(self.current_device)
                local_h = torch.cat((local_h,add), 0)
            if  L > self.cutoff_len:
                local_h = local_h[:self.cutoff_len]
            #print(local_h.shape)
            fixed_features=torch.cat((fixed_features, local_h), 0)
            #print(fixed_features.shape)
            pos += L
        #print(fixed_features.shape)
        fixed_features = fixed_features.view(-1,1,self.cutoff_len,self.feature_dim)
        #print(fixed_features.shape)
        #res_features = self.resnet(fixed_features)
        #print(res_features.shape)
        #caps_features = self.capsnet(fixed_features)


        g_features = self.conv2d_1(fixed_features)
        shapes = g_features.shape     
        g_features = g_features.view(shapes[0],1,shapes[1])
        # print(g_features.shape)

        global_features = torch.tensor([]).reshape(-1, g_features.shape[2]).to(self.current_device)

        cnt = 0
        for idx in chain_idx:
            
            for i in range(seqlens[idx]):
                # print(g_features[cnt].shape)
                global_features = torch.cat((global_features,g_features[cnt]),0)
            cnt += 1
        #print(global_features.shape)

        local_features = torch.tensor([]).reshape(-1,self.feature_dim).to(self.current_device)
        pos = 0
        for idx in chain_idx:
            L = seqlens[idx]
            local_h = single_features[pos:pos+L].view(1, self.feature_dim, L).to(self.current_device)
        #    print(local_h.shape)
            local_h = self.localpool(self.localpad(local_h)).view(L,-1)
        #    print(local_h.shape)
            local_features=torch.cat((local_features, local_h), 0)
            pos = pos + L
        h =  torch.cat((h,local_features,global_features), 1)

        #print(h.shape)
        h = self.DNN1(h)
        h = self.dropout_layer(h)
        h = self.DNN2(h)
        h = self.dropout_layer2(h)
        #print(h.shape)
        #ligand_embed_path='/home/MultiModRLBP-main/ligand_embedding/'+rna_id[0:4]+'_'+c_id.upper()+'_maccs.npy'
        #lig_encode = np.load(ligand_embed_path)
        #lig_encode = torch.tensor(lig_encode).to(torch.float32).to(self.current_device)
        #lig_encode = lig_encode.expand(h.shape[0],-1)
        #print(lig_encode.shape)
        #h = torch.cat((h,lig_encode), 1)
        #print(h.shape)
        h = self.outLayer(h)
        #print(h.shape)
        return h

