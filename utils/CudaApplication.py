import torch


class CudaApplication:
    def __init__(self,size,device):
        self.pointer = torch.zeros(size,size).to(device)
        self.device = device

    def free(self):
        del self.pointer
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

