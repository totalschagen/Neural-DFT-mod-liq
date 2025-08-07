import torch
import random
import gc
import os
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, TensorDataset,Sampler
from collections import OrderedDict

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

class prepared_windows_dataset(Dataset):
    def __init__(self, file_list,len_list, cache_size=2):
        """
        Args:
            file_list: list of file_paths
            len_list: dictionary of lengths for each file, used to create indices
            cache_size: int, number of files to cache in memory
        """
        self.file_list = file_list
        self.file_len =  []
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.indices = []
        self.len_list = len_list


        for file_name in file_list:
            red_file_name=os.path.split(file_name)[-1]  # Get the file name without path
            file_length = len_list[red_file_name]
            self.file_len.append(file_length)
            self.indices.extend([(file_name, i) for i in range(file_length)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_name, window_idx = self.indices[idx]

        # Check if the file is in cache
        if file_name not in self.cache:
            # Load the file and cache it
            data = torch.load(file_name)
            self.cache[file_name] = data

            # If cache exceeds size, remove the oldest entry
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        # Get the cached data
        data = self.cache[file_name]
        window = data['windows'][window_idx]
        label = data['labels'][window_idx]
        return window, label

class prepared_windows_inference_dataset(Dataset):
    def __init__(self, file_list,len_list, cache_size=2):
        """
        Args:
            file_list: list of file_paths
            len_list: dictionary of lengths for each file, used to create indices
            cache_size: int, number of files to cache in memory
        """
        self.file_list = file_list
        self.file_len =  []
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.indices = []
        self.len_list = len_list


        for file_name in file_list:
            red_file_name=os.path.split(file_name)[-1]  # Get the file name without path
            file_length = len_list[red_file_name]
            self.file_len.append(file_length)
            self.indices.extend([(file_name, i) for i in range(file_length)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_name, window_idx = self.indices[idx]

        # Check if the file is in cache
        if file_name not in self.cache:
            # Load the file and cache it
            data = torch.load(file_name)
            self.cache[file_name] = data

            # If cache exceeds size, remove the oldest entry
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        # Get the cached data
        data = self.cache[file_name]
        window = data['windows'][window_idx]
        return window



class prepared_windows_shuffler(Sampler[int]):
    def __init__(self,dataset:prepared_windows_dataset):
        self.dataset = dataset
        self.n_files = len(dataset.file_list)
        self.indices = []
        total_length = 0
        for file_length in dataset.file_len:
            self.indices.append(list(range(total_length, total_length + file_length)))
            total_length += file_length

    def __iter__(self):
        # shuffle file orders
        files = list(range(self.n_files))
        random.shuffle(files)
        # shuffle indices within each file
        for file in files:
            file_indices = self.indices[file][:]
            random.shuffle(file_indices)
            for idx in file_indices:
                yield idx
    
    def __len__(self):
        return len(self.dataset)


class ChunkManager:
    def __init__(self, density_profiles, label_matrices, window_size,stride, chunk_size=5,
                 batch_size=512, shuffle=True, pin_memory=True, num_workers=0):
        """
        Args:
            density_profiles: list of torch.Tensor, each (H, W) on CPU
            label_matrices: list of torch.Tensor, same shape as density_profiles
            window_size: int, size of the sliding window (e.g., 250)
            chunk_size: int, number of matrices per chunk
            batch_size: int, number of windows per batch
        """
        assert len(density_profiles) == len(label_matrices), "Mismatch in matrix list sizes."
        self.density_profiles = density_profiles
        self.label_matrices = label_matrices
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.stride = stride

        self.num_chunks = (len(density_profiles) + chunk_size - 1) // chunk_size

    def _extract_windows_unfold(self, x, y):
        """
        x, y: single matrix pair on GPU, shape (H, W)
        Returns:
            windows: [L, 1, win, win]
            labels: [L]
        """
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        win = self.window_size
        stri = self.stride

        # Unfold to get all windows as columns
        unfolded = F.unfold(x, kernel_size=win, stride=stri)  # [1, win*win, L]
        num_windows = unfolded.shape[-1]

        # Reshape to [L, 1, win, win]
        windows = unfolded.transpose(1, 2).reshape(num_windows, 1, win, win)

        # Compute label positions (center of each window)
        H = x.shape[-2]
        W = x.shape[-1]
        rows = torch.arange(0, H - win + 1, self.stride, device=x.device)
        cols = torch.arange(0, W - win + 1, self.stride, device=x.device)
        ii, jj = torch.meshgrid(rows, cols, indexing='ij')

        center_i = ii + win // 2
        center_j = jj + win // 2
        labels = y[center_i, center_j].reshape(-1)  # [L]

        return windows, labels

    def extract_windows_from_chunk(self, density_chunk, label_chunk):
        all_windows, all_labels = [], []

        for x_cpu, y_cpu in zip(density_chunk, label_chunk):
            # Move to GPU first
            x = x_cpu.cuda(non_blocking=True)
            y = y_cpu.cuda(non_blocking=True)

            windows, labels = self._extract_windows_unfold(x, y)
            all_windows.append(windows)
            all_labels.append(labels)

        return torch.cat(all_windows, dim=0), torch.cat(all_labels, dim=0)

    def get_chunk_loader(self, chunk_idx):
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, len(self.density_profiles))

        density_chunk = self.density_profiles[start:end]
        label_chunk = self.label_matrices[start:end]

        x, y = self.extract_windows_from_chunk(density_chunk, label_chunk)

        dataset = TensorDataset(x, y)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            #pin_memory=self.pin_memory,
            #num_workers=self.num_workers
        )

    def __len__(self):
        return self.num_chunks
class window_chunk_manager():
    def __init__(self, data_list,batch_size=512, shuffle=True):
        """
        Args:
            data_list: list of dictionaries with 'windows' and 'labels' tensors
            batch_size: int, number of windows per batch
        """
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_chunks = len(data_list)
    def get_chunk_loader(self, chunk_idx):
        inputs = self.data_list[chunk_idx]['windows']#.cuda(non_blocking=True)
        targets = self.data_list[chunk_idx]['labels']#.cuda(non_blocking=True)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=False,
            num_workers=0
        )
    def __len__(self):
        return self.num_chunks