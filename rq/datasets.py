import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self, data_path):

        self.data_path = data_path
        
        # Check if it's a txt file (new format) or numpy file (old format)
        if data_path.endswith('.txt'):
            # New format: first column is itemid, second column is multimodal embedding
            self.embeddings = self._load_txt_embeddings(data_path)
        else:
            # Old format: numpy file
            # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
            self.embeddings = np.load(data_path)
        
        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0
            
        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0
            
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")
        
        self.dim = self.embeddings.shape[-1]

    def _load_txt_embeddings(self, data_path):
        """
        Load embeddings from txt file where:
        - First column is itemid
        - Second column is multimodal embedding (assumed to be space-separated floats)
        """
        embeddings = []
        
        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 2:
                    print(f"Warning: Skipping line {line_num} with insufficient columns")
                    continue
                    
                # Skip the first column (itemid) and parse the rest as embedding
                try:
                    embedding = [float(x) for x in parts[1:]]
                    embeddings.append(embedding)
                except ValueError:
                    print(f"Warning: Skipping line {line_num} with invalid float values")
                    continue
        
        return np.array(embeddings, dtype=np.float32)

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)