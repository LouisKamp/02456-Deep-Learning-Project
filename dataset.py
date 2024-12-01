import torch

class FunctionDataset(torch.utils.data.Dataset):
    def __init__(self, solutions_path: str, laplacians_path: str, device: str = "cpu"):
        solutions = torch.load(solutions_path).to(device)
        laplacians = torch.load(laplacians_path).to(device)
        n = solutions.shape[1]
        self.len = len(solutions)

        self.X = torch.zeros((self.len), 1, n,n).to(device)
        self.Y = torch.zeros((self.len), 1, n,n).to(device)

        self.X[:,0] = laplacians
        # self.X[:,1] = solutions
        # self.X[:,1,1:-1,1:-1] = 0

        self.Y[:,0] = solutions
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx].requires_grad_(True), self.Y[idx].requires_grad_(True)