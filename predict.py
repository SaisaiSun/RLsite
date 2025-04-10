import os
import sys
import argparse
import torch
import random
import pickle
from Model import model
from GAT.data_loading import graphloader
from GAT.benchmark import evaluate
from collections import OrderedDict
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

def _set_module(model, submodule_key, submodule_key_new, ernie_embedder):
    print(submodule_key_new)
    setattr(model, submodule_key_new, ernie_embedder)


node_target = ['binding_small-molecule']
node_features = ['dbn']
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="testParams")
    parser.add_argument("--load_path", type=str, default="final_model.pth", help="the model to load")
    parser.add_argument("--test_path", type=str, default="data/test.txt", help="testSet path")
    args = parser.parse_args()

    test_list=[]
    file = open(args.test_path, 'r')
    rna = file.readlines()
    rna = [line.split('\n')[0].lower()+".json" for line in rna if len(line)>=4]
    test_list.extend(rna)
    print(test_list)

    embedder_model = model.RGATEmbedder(infeatures_dim=12, dims=[64, 64])
    classifier_model = model.RGATClassifier(rgat_embedder=embedder_model, ernie_embedder=None, capsnet=None, conv_output=False, return_loss=False, classif_dims=[1])
    #print(classifier_model)
    classifier_model.to(device)

    test_dataset = graphloader.SupervisedDataset(data_path="data/mydata", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",node_features=node_features,node_target=node_target,all_graphs=test_list)
    #test_dataset.setNorm([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 0.34146341463414637])
    test_loader = graphloader.GraphLoader(dataset=test_dataset, split=False, batch_size=1).get_data()

    model_dict = torch.load(args.load_path)
    #classifier_model.load_state_dict(model_dict['model_state_dict'])
    classifier_model.load_state_dict(model_dict)

    fpr, tpr, auc, mcc, precision,  recall, f1_score = evaluate.get_performance(node_target=node_target, node_features=node_features, model=classifier_model,test_loader=test_loader)
    print('We get a performance of :', auc, mcc, precision,  recall, f1_score)
