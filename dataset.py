import torch

class FunctionDataset(torch.utils.data.Dataset):
    def __init__(self, solutions_path: str, laplacians_path: str):
        solutions = torch.load(solutions_path)
        laplacians = torch.load(laplacians_path)
        n = solutions.shape[1]
        self.len = len(solutions)

        self.X = torch.zeros((self.len), 2, n,n)
        self.Y = torch.zeros((self.len), 1, n,n)

        self.X[:,0] = laplacians
        self.X[:,1] = solutions
        self.X[:,1,1:-1,1:-1] = 0

        self.Y[:,0] = solutions
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]