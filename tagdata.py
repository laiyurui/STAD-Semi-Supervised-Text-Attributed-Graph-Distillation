import torch 
import numpy as np 
import pickle
from torch_geometric.data import NeighborSampler
import csv 
import os 
from pathlib import Path

class TextAttributedGraph:

    def __init__(self, args,  run=0):
        
        dataset = args.dataset 

        self.dataset = dataset
        

        self.raw_data = torch.load(f"dataset/{dataset}/processed/geometric_data_processed.pt", weights_only=False)[0]

        if dataset in ['cora', 'citeseer', 'pubmed','elecomp','elephoto', 'dblp', 'bookhis','bookchild', 'products','wikics']:
            self.feats_full = self.raw_data.node_text_feat 
        
        self.nnodes = self.feats_full.shape[0]
        self.edge_index = self.raw_data.edge_index 
        num_edges = self.edge_index.size(1)
        values = torch.ones(num_edges)
        self.edge_attribute = values 
        self.adj_full = torch.sparse_coo_tensor(
            indices=self.edge_index,
            values= values,
            size=(self.nnodes,self.nnodes)
        )
        self.labels_full = self.raw_data.y 
        if dataset in ['cora', 'citeseer','pubmed','products']:
            self.raw_texts = self.raw_data.raw_texts
        elif dataset in ['elecomp', 'elephoto', 'dblp', 'bookhis', 'bookchild', 'wikics']:
            
            file= f'dataset/{dataset}/processed/texts.pkl'
            self.raw_texts= load_pkl_file(file)

        self.nclass = self.labels_full.max().item()+1
        self.class_dict = None
        self.class_dict2 = None
        self.samplers = None

        self.split_dataset(run)
        
        if self.dataset in ['cora','citeseer','pubmed']:
            self.label_name = self.raw_data.category_names 
            unique_categories = []
            for label in self.label_name:
                if label not in unique_categories:
                    unique_categories.append(label)

            category_to_int = {category: idx for idx, category in enumerate(unique_categories)}

            sorted_categories = [category for category, idx in 
                                sorted(category_to_int.items(), key=lambda x: x[1])]
            self.label_space = sorted_categories
            print(self.label_space)
        
        else:
            if os.path.exists(f"dataset/{dataset}/categories.csv"):
                path =f"dataset/{dataset}/categories.csv"
            else:
                path = f"dataset/{dataset}/processed/categories.csv"
            self.label_name = []
            self.create_label_space(path) 
            for y in self.labels_full.cpu().numpy().tolist():
                self.label_name.append(self.label_space[y])
           
        
        idx_file = f"dataset/{dataset}/processed/test_indices_500.pt"
    
        if os.path.exists(idx_file):
            self.test_idx_sel = torch.load(idx_file)
        else:
            if len(self.test_idx) >= 500:
                permuted_idx = self.test_idx[torch.randperm(len(self.test_idx))]
                self.test_idx_sel = permuted_idx[:500]
                torch.save(self.test_idx_sel, idx_file)
            else:
                self.test_idx_sel = self.test_idx
                torch.save(self.test_idx_sel, idx_file)


    def split_dataset(self, run):
        self.load_or_create_indices("dataset/0_splits/"+self.dataset, run)
        self.feats_train = self.feats_full[self.train_idx]
        self.feats_val = self.feats_full[self.val_idx] 
        self.feats_test = self.feats_full[self.test_idx]
        self.texts_train = [self.raw_texts[i] for i in self.train_idx.cpu().numpy().tolist()]
        self.texts_val =  [self.raw_texts[i] for i in self.val_idx.cpu().numpy().tolist()]
        self.texts_test =  [self.raw_texts[i] for i in self.test_idx.cpu().numpy().tolist()]
        self.labels_train = self.labels_full[self.train_idx]
        self.labels_val = self.labels_full[self.val_idx]
        self.labels_test = self.labels_full[self.test_idx] 

        edge_index = self.adj_full.coalesce().indices()
        edge_values = self.adj_full.coalesce().values()

        in_train_set = torch.isin(edge_index[0], self.train_idx) & torch.isin(edge_index[1], self.train_idx)
        sub_indices = edge_index[:, in_train_set]
        sub_values = edge_values[in_train_set]
        unique_nodes, inverse_indices = torch.unique(sub_indices, return_inverse=True)
        mapped_sub_indices = inverse_indices.view(2, -1)
        self.adj_train = torch.sparse_coo_tensor(
            indices=mapped_sub_indices,
            values=sub_values,
            size=(self.train_idx.shape[0], self.train_idx.shape[0])
        ).coalesce()

        in_val_set = torch.isin(edge_index[0], self.val_idx) & torch.isin(edge_index[1], self.val_idx)
        sub_indices = edge_index[:, in_val_set]
        sub_values =  edge_values[in_val_set]
        unique_nodes, inverse_indices = torch.unique(sub_indices, return_inverse=True)
        mapped_sub_indices = inverse_indices.view(2, -1)
        self.adj_val = torch.sparse_coo_tensor(
            indices= mapped_sub_indices,
            values= sub_values,
            size=(self.val_idx.shape[0], self.val_idx.shape[0])
        ).coalesce()

        in_test_set = torch.isin(edge_index[0], self.test_idx) & torch.isin(edge_index[1], self.test_idx)
        sub_indices = edge_index[:, in_test_set]
        sub_values = edge_values[in_test_set]
        unique_nodes, inverse_indices = torch.unique(sub_indices, return_inverse=True)
        mapped_sub_indices = inverse_indices.view(2, -1)
        self.adj_test = torch.sparse_coo_tensor(
            indices=mapped_sub_indices,
            values=sub_values,
            size=(self.test_idx.shape[0], self.test_idx.shape[0])
        ).coalesce()

    def load_or_create_indices(self, save_path, run=0):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        train_path = os.path.join(save_path, f'train_idx_{run}.pt')
        val_path = os.path.join(save_path, f'val_idx_{run}.pt')
        test_path = os.path.join(save_path, f'test_idx_{run}.pt')
        
        if self.dataset not in ['cora', 'pubmed']:
            if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
                self.train_idx = torch.load(train_path)
                self.val_idx = torch.load(val_path)
                self.test_idx = torch.load(test_path)
                return
        
        if self.dataset in ['cora', 'pubmed']:
            if run is None:
                raise ValueError("run parameter is required for cora and pubmed datasets")
            self.train_idx = torch.where(self.raw_data.train_masks[run])[0]
            self.val_idx = torch.where(self.raw_data.val_masks[run])[0]
            self.test_idx = torch.where(self.raw_data.test_masks[run])[0]
        
        elif self.dataset in ['dblp','elecomp', 'elephoto', 'citeseer', 'bookhis', 'bookchild', 'products', 'wikics']:
            all_nodes = torch.arange(self.raw_data.num_nodes)
            all_labels = self.labels_full.cpu().numpy()
            
            unique_classes = np.unique(all_labels)
            new_train_idx = []
            
            for cls in unique_classes:
                cls_nodes = all_nodes[all_labels == cls].cpu().numpy()
                select_num = min(20, len(cls_nodes))
                selected = np.random.choice(cls_nodes, size=select_num, replace=False)
                new_train_idx.extend(selected)
            
            new_train_idx = torch.unique(torch.tensor(new_train_idx, dtype=torch.long))
            
            remaining_nodes = torch.tensor([n for n in all_nodes if n not in new_train_idx], dtype=torch.long)
            remaining_labels = all_labels[remaining_nodes.cpu().numpy()]
            
            val_idx = []
            remaining_per_cls = {cls: remaining_nodes[remaining_labels == cls].cpu().numpy() 
                                for cls in unique_classes}
            
            total_remaining = len(remaining_nodes)
            for cls in unique_classes:
                cls_count = len(remaining_per_cls[cls])
                if total_remaining == 0:
                    cls_val_num = 0
                else:
                    cls_val_num = int(round(500 * cls_count / total_remaining))
                cls_val_num = min(cls_val_num, cls_count)
                val_idx.extend(np.random.choice(remaining_per_cls[cls], size=cls_val_num, replace=False))
            
            val_idx = np.array(val_idx)
            if len(val_idx) < 500:
                remaining_after_val = [n for n in remaining_nodes.cpu().numpy() if n not in val_idx]
                val_idx = np.concatenate([val_idx, 
                                        np.random.choice(remaining_after_val, 
                                                        size=500 - len(val_idx), 
                                                        replace=False)])
            elif len(val_idx) > 500:
                val_idx = np.random.choice(val_idx, size=500, replace=False)
            
            new_val_idx = torch.tensor(val_idx, dtype=torch.long)
            new_test_idx = torch.tensor([n for n in remaining_nodes.cpu().numpy() 
                                        if n not in new_val_idx], dtype=torch.long)

            self.train_idx = new_train_idx
            self.val_idx = new_val_idx
            self.test_idx = new_test_idx
        
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.train_idx, train_path)
        torch.save(self.val_idx, val_path)
        torch.save(self.test_idx, test_path)

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]
        else:
            sizes = [15,10,5]

        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.train_idx)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=8, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out

    def create_label_space(self,csv_file_path):
        if self.dataset == 'elephoto':
            self.label_space = ['Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 
                               'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        elif self.dataset == 'elecomp':
            self.label_space = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 
                               'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts']
        elif self.dataset == 'bookchild':
            self.label_space = ['Literature & Fiction','Animals','Growing Up & Facts of Life','Humor','Cars Trains & Things That Go','Fairy Tales Folk Tales & Myths',
                               'Activities Crafts & Games','Science Fiction & Fantasy','Classics','Mysteries & Detectives','Action & Adventure','Geography & Cultures','Education & Reference',
                               'Arts Music & Photography','Holidays & Celebrations','Science Nature & How It Works','Early Learning','Biographie','History',"Children's Cookbooks", 'Religions',
                               'Sports & Outdoors','Comics & Graphic Novels','Computers & Technology'
                               ]
        elif self.dataset == 'bookhis':
            self.label_space = ['World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa', 
                                'Ancient Civilizations', 'Middle East', 'Historical Study & Educational Resources', 'Australia & Oceania', 'Arctic & Antarctica']
        elif self.dataset == 'wikics':
            self.label_space = ['Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 
                               'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics']
        elif self.dataset == 'dblp':
            self.label_space = ['Database','Data Mining','AI','Information Retrieval']

        elif self.dataset == 'cora':
            self.label_space = ['Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods']
        elif self.dataset == 'citeseer':
            self.label_space=[ 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)']
        elif self.dataset == 'pubmed':
            self.label_space = ['Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']

        return 
       

def load_pkl_file(file_path):
    data = np.load(file_path, allow_pickle=True)

    result_dict = {}
    for key in data:
        value = data[key]
        
        if isinstance(value, np.ndarray):
            result_dict[key] = value.tolist()
        elif isinstance(value, bytes):
            try:
                result_dict[key] = pickle.loads(value)
            except pickle.UnpicklingError:
                result_dict[key] = value.decode('utf-8', errors='ignore')
        else:
            result_dict[key] = value

    data.close()

    result = list(result_dict.values()) 
    ll =0
    for l in result[0]:
        ll += len(l)
    
    return result[0][0]
    

    
