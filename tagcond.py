import re 
import numpy as np
import torch
import torch.nn.functional as F 
from sklearn.cluster import KMeans, MiniBatchKMeans
from models.GNN import GCN, SelfTrainingGNN
from deep_robust_utils import normalize_adj_tensor
from utils_gdd import using_mlp, clst_condense_adj, clst_condense_feat, clst_condense_label,regenerate_cluster_labels
from sentence_transformers import SentenceTransformer
import torch, re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from keys import QWEN_KEY 
from prompt import *

dataset_sys_mapping = {
    "cora": SYS_CORA,
    "citeseer": SYS_CIESEER,
    "pubmed": SYS_PUBMED,
    "dblp": SYS_DBLP,
    "elecomp": SYS_ELECOMP,
    "elephoto": SYS_ELEPHOTO,
    "bookhis": SYS_BOOKHIS,
    "bookchild": SYS_BOOKCHILD,
    "wikics": SYS_WIKICS
}


GNNs = ['gcn', 'gat']
LMs = ['']
LLM_GNN = ['TAPE', 'One4All']


class TAGCond:
    def __init__(self, data, args, device='cuda',llm_id ="Qwen/Qwen3-1.7B",  **kwargs):
        self.data = data
        self.args = args
        self.device = device

        self.ori_node_num =  data.feats_full.shape[0]
        print(data.feats_train.shape)

        n = int(data.feats_train.shape[0] * args.reduction_rate)
        d = data.feats_train.shape[1]
        
        self.nnodes_syn = n
        self.d = d 
        self.args = args

        self.sbert = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.sbert_hidden_size= self.sbert.get_sentence_embedding_dimension()
        for module in self.sbert.modules():
            for param in module.parameters(recurse=False):  
                param.requires_grad = False
        self.sbert.eval() 

        print('adj_syn:', (n,n), 'feat_syn:', (n,d))
    
    def get_model(self, model_name, mode='pre'):
        model = None 
        if mode == 'pre':
            nhid=self.args.phidden
            nlayers=self.args.pnlayers
            dropout=self.args.pdropout
        elif mode == 'post':
            nhid=self.args.pohidden
            nlayers=self.args.ponlayers
            dropout=self.args.podropout
        elif mode == 'eval':
            nhid=self.args.ehidden
            nlayers=self.args.enlayers
            dropout=self.args.edropout

        
        if model_name == 'gcn':
            model = GCN(nfeat=self.nfeats, nhid=nhid, nclass=self.nclasses, 
                        nlayers=nlayers, dropout=dropout,device=self.device).to(self.device)
        elif model_name == 'stgnn':
            model = SelfTrainingGNN(input_dim=self.nfeats, hidden_dim=nhid,num_classes=self.nclasses,alpha=self.args.alpha, k=self.args.propnum, mode=self.args.stmode).to(self.device)

        return model 

    def clean(self, docs):
        def _clean(txt: str) -> str:
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = txt.translate(str.maketrans("", "", string.punctuation))
            txt = re.sub(r"\d+", " ", txt)
            return re.sub(r"\s+", " ", txt.lower()).strip()
        return [_clean(d) for d in docs]
    
    def tfidf_topk(self, docs, k):
        vect = TfidfVectorizer(max_features=50_000, ngram_range=(1, 1), stop_words="english")
        X = vect.fit_transform(docs)
        id2word = {i: w for w, i in vect.vocabulary_.items()}
        scores = X.sum(axis=0).A1
        top_idx = scores.argsort()[::-1][:k]
        return [id2word[i] for i in top_idx]

    def sbert_topk(self, words_2k, centroid, final_k):
        embs = self.sbert.encode(words_2k, convert_to_tensor=True, device=self.device)
        distances = torch.norm(embs - centroid, dim=1) 
        final_k = min(final_k, distances.shape[0])
        top_idx = distances.topk(final_k, largest=False).indices.cpu().tolist()
        weight =  (1./(1e-4+distances[top_idx])).cpu().numpy().tolist()
        return [words_2k[i] for i in top_idx],weight
    
    def extract_keywords(self, docs, centroid, kk, mode='normal'):
        docs = self.clean(docs)
        import random 
        if mode=='normal':
            words_2k = self.tfidf_topk(docs, 2 * kk)
            return self.sbert_topk(words_2k, centroid, kk)
        else:
            tfidf_vocab = self.tfidf_topk(docs, k=50_000)
            if kk > len(tfidf_vocab):
                random_words = tfidf_vocab.copy()
            else:
                random_words = random.sample(tfidf_vocab, kk)
            weights = [1.0 for _ in range(len(random_words))]
            return random_words, weights

    def extract_keysentence(self, text_list, centroid, total_max_words):

        filter_prefixes = ["feature node"]
        valid_texts = []
        for text in text_list:
            text = text.strip()
            if not text:
                continue
            for prefix in filter_prefixes:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
                    break
            if text and text.lower() not in [p.lower() for p in filter_prefixes]:
                valid_texts.append(text)
        
        if not valid_texts:
            return [], []

        full_text = "\n".join(valid_texts)
        sentences = re.split(r'[.\n!?;:\t]', full_text)
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        min_word_count = 3
        filtered_sentences = []
        for sent in sentences:
            if any(p.lower() == sent.lower() for p in filter_prefixes):
                continue
            if len(sent.split()) < min_word_count:
                continue
            filtered_sentences.append(sent)
        
        if not filtered_sentences:

            filtered_sentences = valid_texts[:3]
            if not filtered_sentences:
                return [], []
        
        model = self.sbert
        sentence_embeddings = model.encode(
            filtered_sentences,  
            convert_to_tensor=True,
            device=self.device,  
            normalize_embeddings=True
        )
        
        distances = torch.norm(sentence_embeddings - centroid, dim=1)  
        sorted_indices = torch.argsort(distances).cpu().tolist()  

        sorted_sentences = []
        seen_sents = set() 
        for i in sorted_indices:
            sent = filtered_sentences[i]
            dist = distances[i].cpu().item()
            sent_lower = sent.lower()
            if sent_lower not in seen_sents:
                seen_sents.add(sent_lower)
                sorted_sentences.append((sent, dist))
        
        selected_sentences = []
        selected_weights = []  
        current_total_words = 0
        
        for sent, dist in sorted_sentences:
            word_count = len(sent.split())
            
            if current_total_words + word_count <= total_max_words:
                selected_sentences.append(sent)
                weight = 1.0 / (1e-4 + dist)
                selected_weights.append(weight)
                current_total_words += word_count
            if current_total_words >= total_max_words:
                break
        
        if not selected_sentences and filtered_sentences:
            first_sent = filtered_sentences[0]
            if len(first_sent.split()) <= total_max_words:
                selected_sentences.append(first_sent)
                selected_weights.append(1.0)
        
        return selected_sentences, selected_weights

    def graph_ST_clustering(self):
        eye = using_mlp(self.data,self.device,signal=True)
        adj_norm = using_mlp(self.data,self.device,signal=False) 
        
        train_doc_indices = self.data.train_idx.to(self.device)  
        val_doc_indices = self.data.val_idx.to(self.device)
        test_doc_indices = self.data.test_idx.to(self.device)
        doc_labels =  self.data.labels_full.to(self.device)
        test_labels = doc_labels[test_doc_indices]

        GNN_best_val_acc = 0.
        GNN_best_test_acc = 0. 
        MLP_best_val_acc = 0. 
        MLP_best_test_acc = 0. 
        
        if self.args.compare != False:
            best_val_accuracy1 = 0. 
            model1 = self.get_model('gcn',mode='pre')
            optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.args.plr, weight_decay=self.args.pweight_decay)
            for epoch in range(self.args.pepochs):
                model1.train()
                optimizer1.zero_grad()
                out = model1(self.data.feats_full.to(self.device), adj_norm)
                train_loss1 = F.cross_entropy(out[train_doc_indices], doc_labels[train_doc_indices])
                train_loss1.backward()
                optimizer1.step()
                model1.eval()
                with torch.no_grad():  
                    out = model1(self.data.feats_full.to(self.device), adj_norm)
                    val_preds1 = out[val_doc_indices].argmax(dim=1)
                    val_labels1 = doc_labels[val_doc_indices]
                    val_accuracy1 = (val_preds1 == val_labels1).float().mean()
                if val_accuracy1.item() > best_val_accuracy1:
                    best_val_accuracy1 = val_accuracy1.item()
                    best_test_accuracy1 = (out[test_doc_indices].argmax(dim=1) == test_labels).float().mean()

            GNN_best_val_acc= best_val_accuracy1
            GNN_best_test_acc = best_test_accuracy1.item()
            print("GCN Best Val Acc:", GNN_best_val_acc)
            print("GCN Best Test Acc:", GNN_best_test_acc)

            best_val_accuracy1 = 0. 
            model1 = self.get_model('gcn',mode='pre')
            optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.args.plr, weight_decay=self.args.pweight_decay)
            for epoch in range(self.args.pepochs):
                model1.train()
                optimizer1.zero_grad()
                out = model1(self.data.feats_full.to(self.device), eye)
                train_loss1 = F.cross_entropy(out[train_doc_indices], doc_labels[train_doc_indices])
                train_loss1.backward()
                optimizer1.step()
                model1.eval()
                with torch.no_grad():  
                    out = model1(self.data.feats_full.to(self.device), eye)
                    val_preds1 = out[val_doc_indices].argmax(dim=1)
                    val_labels1 = doc_labels[val_doc_indices]
                    val_accuracy1 = (val_preds1 == val_labels1).float().mean()
                if val_accuracy1.item() > best_val_accuracy1:
                    best_val_accuracy1 = val_accuracy1.item()
                    best_test_accuracy1 = (out[test_doc_indices].argmax(dim=1) == test_labels).float().mean()
            MLP_best_val_acc = best_val_accuracy1
            MLP_best_test_acc = best_test_accuracy1.item()
                 
            print("MLP Best Val Acc:", MLP_best_val_acc)
            print("MLP Best Test Acc:", MLP_best_test_acc) 

        from utils_gdd import indices_to_masks
        train_mask , val_mask, test_mask, unlabeled_mask = indices_to_masks(self.nnodes, train_doc_indices, val_doc_indices, test_doc_indices)
        best_pre_val_accuracy = 0. 
        stgnn = self.get_model('stgnn',mode='pre')
        best_logits, best_adj, _ , best_val_acc, best_test_acc = stgnn.self_train(self.data.feats_full.to(self.device).clone(), adj_norm.clone(), doc_labels.clone(), train_mask.clone(), val_mask.clone(), test_mask.clone(), unlabeled_mask.clone(), 
            epochs_per_stage=600, initial_ratio=0.1, ratio_step=0.10,  pretrain_stages=self.args.ps,  
            num_stages=self.args.num_stages, w_gnn=self.args.w_gnn, w_mlp=self.args.w_mlp, w_fused=self.args.w_fused)
        STGNN_best_val_acc = best_val_acc
        STGNN_best_test_acc = best_test_acc
        print("STGCN Best Val Acc:", STGNN_best_val_acc)
        print("STGCN Best Test Acc:", STGNN_best_test_acc)
        
        raw_text = self.data.raw_texts
        
        n_clusters=self.nnodes_syn

        cluster_method = getattr(self.args, "cluster_method", "kmeans").lower()
        cluster_features = best_logits.cpu().detach().numpy()
        if cluster_method == "minibatch-kmeans":
            batch_size = min(cluster_features.shape[0], getattr(self.args, "cluster_batch_size", 100))
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                random_state=getattr(self.args, "seed", None),
            )
        else:
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=getattr(self.args, "seed", None),
            )
        clusters = clusterer.fit_predict(cluster_features)
        clusters, n_final_clusters = regenerate_cluster_labels(clusters, best_logits, n_global_clusters=n_clusters )
        
        attributes_condensed = clst_condense_feat(clusters, n_clusters, self.data.feats_full.cpu().numpy())

        attributes_condensed = attributes_condensed.to(self.device)

        adj_condensed, cluster_edge_indices, cluster_edge_weights = clst_condense_adj(best_adj,n_clusters,self.nnodes,clusters,self.device)

        labels_condensed = clst_condense_label(n_clusters,best_logits,clusters,self.device)

        print("Finish clustering-based condensation.")

        key_words_list =[]
        key_words_weights_list = []
        cluster_to_texts = {}
        for cluster_label, text in zip(clusters, raw_text):
            if cluster_label not in cluster_to_texts:
                cluster_to_texts[cluster_label] = []
            cluster_to_texts[cluster_label].append(text)

        self.cluster_to_texts = cluster_to_texts

        input_mode = 'normal'
        for i in range(self.nnodes_syn):
            cluster_texts = cluster_to_texts.get(i, [])
            key_words, key_words_weights = self.extract_keywords(cluster_texts, attributes_condensed[i], kk=self.args.word_lim, mode=input_mode)
            key_words_list.append(key_words)
            key_words_weights_list.append(key_words_weights)

        return attributes_condensed, adj_condensed, labels_condensed, key_words_list, best_logits, \
            GNN_best_val_acc, GNN_best_test_acc, MLP_best_val_acc, MLP_best_test_acc, STGNN_best_val_acc, STGNN_best_test_acc, key_words_weights_list
    

    def LLM_synthesis_api(self,attributes_condensed, adj_condensed, labels_condensed,keywords, keywords_weights, run=0):
        import random
 
        condense_best_val_accuracy, condense_best_test_accuracy =  self.test_with_val(attributes_condensed, adj_condensed, labels_condensed)
            
        print("Condense Best Val Acc:", condense_best_val_accuracy)
        print("Condense Best Test Acc:", condense_best_test_accuracy)
        llm_text_val_acc0, llm_text_test_acc0 = 0., 0.
        
        label_space_str = self.data.label_space  
        self.label_to_idx = {label: idx for idx, label in enumerate(label_space_str)}
        
        self.args.qwen_api_key =  QWEN_KEY
        client = None if getattr(self.args, "notext", 0) == 1 else OpenAI(api_key=self.args.qwen_api_key, base_url=self.args.qwen_api_base_url)

        texts_syn_dict = {}
        labels_condensed_new_list = []

        for i in range(self.nnodes_syn):
            kw_list_temp = keywords[i]
            kw_weight = keywords_weights[i] 
            
            center_label1 = label_space_str[labels_condensed[i].item()]
            
            output_all = []

            for j in range(self.args.dnum):
                center_key =  ', '.join(kw_list_temp)
                print("The center key len is",len(center_key))
                
                user_prompt = user_prompt_template_text.format( center_key=center_key,
                    center_label1=center_label1,
                    tok_lim=self.args.tok_lim,
                )
                system_prompt = system_prompt_template_text.format(dataset_name = self.args.dataset, sys_dataset = dataset_sys_mapping[self.args.dataset])

                if getattr(self.args, "notext", 0) == 1:
                    output = center_key
                else:
                    try:
                        response = client.chat.completions.create(
                            model=self.args.qwen_model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=1.2,  
                            top_p=0.85,  
                            max_tokens=self.args.tok_lim ,     
                            extra_body={"enable_thinking": False},
                        )
                        output = response.choices[0].message.content.strip()
                    except:
                        output = center_key
                
                try:
                    print(output)
                except UnicodeEncodeError:
                    safe_output = output.encode('cp1252', errors='replace').decode('cp1252')
                    print("Output (special chars replaced):", safe_output)

                output_all.append(output)
            
            texts_syn_dict[i]= output_all
        
        for idx in range(self.nnodes_syn):
            labels_condensed_new_list.append(label_space_str[labels_condensed[idx].item()])

        labels_syn = torch.tensor(
                    [self.label_to_idx[label] for label in labels_condensed_new_list],
                    dtype=torch.long,
                    device=self.device
                )
        
        adj_syn = adj_condensed

        attributes_syn, best_texts_syn, align_scores= self.candidates_choose(texts_syn_dict, adj_syn,labels_syn, candidates_num=self.args.cdnum, candidates_num1=self.args.cdnum1)
        
        llm_text_val_acc0 ,llm_text_test_acc0  = self.test_with_val(attributes_syn, adj_syn, labels_syn)
        print("W condensed graph, LLM Text Best Val Acc:", llm_text_val_acc0)
        print("W condensed graph, LLM Text Best Test Acc:", llm_text_test_acc0)

        if self.args.save_file==1:
            import os
            save_dir = f"./results/ours/{self.args.qwen_model_name}/{self.args.dataset}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
             
            def save_data(data, filename):
                try:
                    save_path = os.path.join(save_dir, filename)
                    if isinstance(data, torch.Tensor):
                        torch.save(data.cpu(), save_path)
                    elif isinstance(data, list) and all(isinstance(x, str) for x in data):
                        with open(save_path, 'w', encoding='utf-8', errors='replace') as f:
                            for idx, text in enumerate(data):
                                f.write(f"=== Text {idx} ===\n{text}\n\n")
                    else:
                        torch.save(data, save_path)
                    print(f"Successfully saved: {save_path}")
                except Exception as e:
                    print(f"Failed to save {filename}: {e}")
            
            save_data(attributes_condensed, f"attributes_condensed_{self.args.reduction_rate}_run_{run}.pt")
            save_data(adj_condensed, f"adj_condensed_{self.args.reduction_rate}_run_{run}.pt")
            save_data(labels_condensed, f"labels_condensed_run_{self.args.reduction_rate}_{run}.pt")
            save_data(best_texts_syn, f"best_texts_syn_run_{self.args.reduction_rate}_{run}.txt")  
            save_data(attributes_syn, f"attributes_syn_run_{self.args.reduction_rate}_{run}.pt")
            save_data(adj_syn, f"adj_syn_run_{self.args.reduction_rate}_{run}.pt")
            save_data(labels_syn, f"labels_syn_run_{self.args.reduction_rate}_{run}.pt")
            
            acc_results = {
                "run": run,
                "dataset": self.args.dataset,
                "condense_best_val_acc": condense_best_val_accuracy,
                "condense_best_test_acc": condense_best_test_accuracy,
                "llm_text_val_acc0": llm_text_val_acc0,
                "llm_text_test_acc0": llm_text_test_acc0,
            }
            save_data(acc_results, f"accuracy_results_{self.args.reduction_rate}_run_{run}.pt")

        return condense_best_val_accuracy, condense_best_test_accuracy, llm_text_val_acc0, llm_text_test_acc0


    def candidates_choose(self, text_synonyms_dict, adj_syn,label_syn, candidates_num=100, candidates_num1 = 5 ):
        import random 
        candidate_feature_list = []
        candidates_texts_list = []
        
        for i in range(candidates_num):
            current_candidate_feats = []
            current_candidate_texts = []
            
            with torch.no_grad():
                for cluster_idx, syn_texts in sorted(text_synonyms_dict.items()):
                    if i==0:
                        selected_text = syn_texts[0] if isinstance(syn_texts, list) else syn_texts
                    else:
                        selected_text = random.choice(syn_texts) if isinstance(syn_texts, list) else syn_texts
                    current_candidate_texts.append(selected_text)
                    
                    text_embedding = self.sbert.encode(selected_text, convert_to_tensor=True).unsqueeze(0)
                    current_candidate_feats.append(text_embedding)
                
                candidate_feat = torch.cat(current_candidate_feats, dim=0)
            
            candidate_feature_list.append(candidate_feat)
            candidates_texts_list.append(current_candidate_texts)

        
        from utils_gdd import smooth_twice, subsample_sinkhorn
        
        align_scores = [] 
        for idx, candidate in enumerate(candidate_feature_list):
            try:
                Z_original = smooth_twice(self.data.adj_full, self.data.feats_full)
                Z_synthetic = smooth_twice(adj_syn, candidate)

                l2_norm_original = Z_original.norm(dim=1, keepdim=True) + 1e-8
                Z_original = Z_original / l2_norm_original 

                l2_norm_synthetic = Z_synthetic.norm(dim=1, keepdim=True) + 1e-8
                Z_synthetic = Z_synthetic / l2_norm_synthetic 

                align_score = subsample_sinkhorn(Z_original, Z_synthetic, sub=2000, eps=0.05)
                align_scores.append(align_score.item() if torch.is_tensor(align_score) else align_score)
                
            except Exception as e:
                print(f"Error calculating align score for candidate {idx}: {str(e)}")
                align_scores.append(float('inf'))
        
        min_align_idx = torch.argmin(torch.tensor(align_scores)).item()
        best_candidate_feat = candidate_feature_list[min_align_idx]
        best_candidate_texts = candidates_texts_list[min_align_idx]

        score_feat_text = list(zip(align_scores, candidate_feature_list, candidates_texts_list))
        score_feat_text_sorted = sorted(score_feat_text, key=lambda x: x[0])
        top_candidates = score_feat_text_sorted[:candidates_num1]

        best_val_acc = 0.0
        final_best_feat = None
        final_best_texts = None
        final_best_test_acc = 0.0

        for idx, (score, feat, texts) in enumerate(top_candidates):
            try:
                val_acc, test_acc = self.test_with_val(
                    attributes_syn=feat.to(self.device),  
                    adj_syn=adj_syn.to(self.device),
                    labels_syn=label_syn
                )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_best_feat = feat
                    final_best_texts = texts
                    final_best_test_acc = test_acc
            except Exception as e:
                print(f"Continue")

        if final_best_feat is None:
            final_best_feat = best_candidate_feat
            final_best_texts = best_candidate_texts
            
        return final_best_feat, final_best_texts, align_scores

    def test_with_val(self, attributes_syn, adj_syn, labels_syn ):
        train_doc_indices = self.data.train_idx.to(self.device)  
        val_doc_indices = self.data.val_idx.to(self.device)
        test_doc_indices = self.data.test_idx.to(self.device)
        doc_labels =  self.data.labels_full.to(self.device)
        test_labels = doc_labels[test_doc_indices]
        adj_norm = normalize_adj_tensor(self.data.adj_full,sparse=True).to(self.device)

        best_val_accuracy1 = 0. 
        model1 = self.get_model(self.args.eval_model).to(self.device)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=self.args.elr, weight_decay=self.args.eweight_decay)
        
        for epoch in range(self.args.pepochs):
            model1.train()
            optimizer1.zero_grad()
            out = model1(attributes_syn, adj_syn)
            train_loss1 = F.cross_entropy(out, labels_syn)
            train_loss1.backward()
            
            optimizer1.step()
            model1.eval()
            with torch.no_grad():  
                out = model1(self.data.feats_full.to(self.device), adj_norm)
                val_preds1 = out[val_doc_indices].argmax(dim=1)
                val_labels1 = doc_labels[val_doc_indices]
                val_accuracy1 = (val_preds1 == val_labels1).float().mean()

            if val_accuracy1.item() > best_val_accuracy1:
                best_val_accuracy1 = val_accuracy1.item()
                best_test_accuracy1 = (out[test_doc_indices].argmax(dim=1) == test_labels).float().mean()
        
        return best_val_accuracy1 , best_test_accuracy1.item()
         
            
    def train(self):
        import time
        t1 = time.time() 

        GNN_best_val_acc_list =[]
        GNN_best_test_acc_list =[]
        MLP_best_val_acc_list =[]
        MLP_best_test_acc_list =[]
        STGNN_best_val_acc_list =[]
        STGNN_best_test_acc_list =[]

        condense_best_val_acc_list = []
        condense_best_test_acc_list =[]

        LLM_best_text_val_acc_list = []
        LLM_best_text_test_acc_list=[]

        LLM_best_text_val_acc_list1 = []
        LLM_best_text_test_acc_list1=[]

        LLM_best_text_val_acc_list2 = []
        LLM_best_text_test_acc_list2=[]
        for run in range(self.args.pruns):
            self.data.split_dataset(run)
            data = self.data 
            self.nfeats = data.feats_full.shape[-1]
            self.nnodes = data.feats_full.shape[0]
            self.nclasses = data.labels_full.max().item()+1
            self.nclusts = self.nnodes_syn

            attributes_condensed, adj_condensed, labels_condensed, key_words, best_logits, \
            GNN_best_val_acc, GNN_best_test_acc, MLP_best_val_acc, \
            MLP_best_test_acc, STGNN_best_val_acc, STGNN_best_test_acc,keywords_weights\
            = self.graph_ST_clustering()

            GNN_best_val_acc_list.append(GNN_best_val_acc)
            GNN_best_test_acc_list.append(GNN_best_test_acc)
            MLP_best_val_acc_list.append(MLP_best_val_acc)
            MLP_best_test_acc_list.append(MLP_best_test_acc)
            STGNN_best_val_acc_list.append(STGNN_best_val_acc)
            STGNN_best_test_acc_list.append(STGNN_best_test_acc)

            condense_best_val_acc, condense_best_test_acc, llm_text_val_acc0,llm_text_test_acc0  = self.LLM_synthesis_api(attributes_condensed, 
                                                                                adj_condensed, labels_condensed, 
                                                                                key_words, keywords_weights, run)
            
            
            condense_best_val_acc_list.append(condense_best_val_acc)
            condense_best_test_acc_list.append(condense_best_test_acc)

            LLM_best_text_val_acc_list.append(llm_text_val_acc0)
            LLM_best_text_test_acc_list.append(llm_text_test_acc0) 

        
        GNN_best_val_acc_arr = np.array(GNN_best_val_acc_list)
        MLP_best_val_acc_arr = np.array(MLP_best_val_acc_list)
        STGNN_best_val_acc_arr = np.array(STGNN_best_val_acc_list)
        condense_best_val_acc_arr = np.array(condense_best_val_acc_list)
        LLM_best_text_val_acc_arr = np.array(LLM_best_text_val_acc_list)

        GNN_best_test_acc_arr = np.array(GNN_best_test_acc_list)
        MLP_best_test_acc_arr = np.array(MLP_best_test_acc_list)
        STGNN_best_test_acc_arr = np.array(STGNN_best_test_acc_list)
        condense_best_test_acc_arr= np.array(condense_best_test_acc_list)
        LLM_best_text_test_acc_arr = np.array(LLM_best_text_test_acc_list)

        
        def print_mean_std(arr, metric_name):
            mean = np.mean(arr)
            std = np.std(arr) 
            print(f"{metric_name} - Mean: {mean:.4f}, Std: {std:.4f}")

        print("=== Validation Accuracy Statistics ===")
        print_mean_std(GNN_best_val_acc_arr, "GNN Best Val Acc")
        print_mean_std(MLP_best_val_acc_arr, "MLP Best Val Acc")
        print_mean_std(STGNN_best_val_acc_arr, "STGNN Best Val Acc")
        print_mean_std(condense_best_val_acc_arr, "Condense Best Val Acc")
        print_mean_std(LLM_best_text_val_acc_arr, "W condensed graph, Text Distill via LLM Best Val Acc")

        print("\n=== Test Accuracy Statistics ===")
        print_mean_std(GNN_best_test_acc_arr, "GNN Best Test Acc")
        print_mean_std(MLP_best_test_acc_arr, "MLP Best Test Acc")
        print_mean_std(STGNN_best_test_acc_arr, "STGNN Best Test Acc")
        print_mean_std(condense_best_test_acc_arr, "Condense Best Test Acc")
        print_mean_std(LLM_best_text_test_acc_arr, "W condensed graph, Text Distill via LLM Best Test Acc")
        return 
 
