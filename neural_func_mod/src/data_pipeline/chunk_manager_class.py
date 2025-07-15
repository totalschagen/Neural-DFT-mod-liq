import torch
from torch.utils.data import Dataset
import random

class SlidingWindowDataset(Dataset):
    def __init__(self, density_profiles,c1_matrix, window_size,):
        ''' density profiles: list of torch.tensor, of continuous shape
            window_size: int, size of the sliding window
        '''
        assert (len(density_profiles) == len(c1_matrix)), "Density profiles and c1_matrix must have the same length"
        self.density_profiles = density_profiles
        self.c1_matrix = c1_matrix
        self.window_size = window_size
        self.indices = []
        #self.density_profiles=[mat.cuda(non_blocking=True) for mat in density_profiles]
        #self.c1_matrix = [mat.cuda(non_blocking=True) for mat in c1_matrix]

        #precompute indices for sliding windows
        for rho_id,rho in enumerate(self.density_profiles):
            if len(rho) < window_size:
                continue
            N = rho.size(0)
            for i in range(N- window_size + 1):
                for j in range(N-window_size + 1):
                    self.indices.append((rho_id,i,j))
        # move matrices to GPU if available
        if torch.cuda.is_available():
            self.density_profiles = [mat.cuda(non_blocking=True) for mat in density_profiles]
            self.c1_matrix = [mat.cuda(non_blocking=True) for mat in c1_matrix]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx): 
        rho_id, i, j = self.indices[idx]

        x = self.density_profiles[rho_id]
        y = self.c1_matrix[rho_id]
        window = x[i:i+self.window_size, j:j+self.window_size]
        window = window.unsqueeze(0)  # Add a channel dimension

        center_i = i + self.window_size // 2
        center_j = j + self.window_size // 2
        value = y[center_i, center_j]
        return window,value

    