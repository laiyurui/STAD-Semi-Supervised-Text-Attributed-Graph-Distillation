import torch
import argparse
import os
from tagcond import TAGCond

from utils_gdd import * 
from tagdata import TextAttributedGraph 

def main(args):
    print(args)
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    data = TextAttributedGraph(args)

    print(device)
    import time 
    t1 = time.time()
    if args.gdd_method == 'TAGCond':
        agent = TAGCond(data,args, device)
    t2 = time.time() 
    agent.train() 
    print(f"time is {t2-t1}")

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--normalize_features', type=bool, default=False) 
    parser.add_argument('--gdd_method',type = str, default='TAGCond')
    parser.add_argument('--reduction_rate', type=float, default=1,help="the ratio of syn data/ori train data")
    parser.add_argument('--num_stages',type = int, default=10)
    parser.add_argument('--alpha',type = float, default=0.5)
    parser.add_argument('--propnum',type = int, default=2)
    parser.add_argument('--pretrain_model',type = str, default='gcn')
    parser.add_argument('--pnlayers', type=int, default=2)
    parser.add_argument('--pruns', type=int, default=5)
    parser.add_argument('--phidden', type=int, default=64)
    parser.add_argument('--pweight_decay', type=float, default=0.0005) 
    parser.add_argument('--plr', type=float, default=0.01)
    parser.add_argument('--pdropout', type=float, default=0.5)
    parser.add_argument('--pepochs', type=int, default=600)
    parser.add_argument('--w_gnn', type=float, default=1.0)
    parser.add_argument('--w_mlp', type=float, default=1.0)
    parser.add_argument('--w_fused', type=float, default=1.0)
    parser.add_argument('--compare', type=bool, default=False)
    parser.add_argument('--stmode', type=str, default='gcnmlp')
    parser.add_argument('--ps', type=int, default=0)
    parser.add_argument("--qwen_model_name", type=str, default= "qwen3-max")
    parser.add_argument("--qwen_api_key", type=str, default=os.getenv("QWEN_KEY"),
                        help="Qwen API key (Aliyun DashScope, can set via environment variable)")
    parser.add_argument("--qwen_api_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="Qwen API base URL (default: Aliyun DashScope)")
    parser.add_argument('--dnum', type=int, default=3)
    parser.add_argument('--cdnum', type=int, default=100)
    parser.add_argument('--cdnum1', type=int, default=5)
    parser.add_argument('--word_lim', type=int, default=256)
    parser.add_argument('--tok_lim', type=int, default=256)
    parser.add_argument('--notext', type=int, default=0)
    parser.add_argument('--cluster_method', type=str, default='kmeans',
                        choices=['kmeans', 'minibatch-kmeans'])
    parser.add_argument('--cluster_batch_size', type=int, default=100)
    parser.add_argument('--ponlayers', type=int, default=2)
    parser.add_argument('--pohidden', type=int, default=64)
    parser.add_argument('--poweight_decay', type=float, default=0.0005) 
    parser.add_argument('--polr', type=float, default=0.01)
    parser.add_argument('--podropout', type=float, default=0.0)
    parser.add_argument('--poepochs', type=int, default=200)
    parser.add_argument('--eval_model',type = str, default='gcn') 
    parser.add_argument('--eruns', type=int, default=5)
    parser.add_argument('--enlayers', type=int, default=2)
    parser.add_argument('--ehidden', type=int, default=64)
    parser.add_argument('--eweight_decay', type=float, default=0.0005) 
    parser.add_argument('--elr', type=float, default=0.01)
    parser.add_argument('--edropout', type=float, default=0.0)
    parser.add_argument('--eepochs', type=int, default=600)
    parser.add_argument('--save_file', type=int, default=0)
    args = parser.parse_args()
    main(args)
