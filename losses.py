import torch

class HistogramLoss(torch.nn.Module):
    def __init__(self, bins, device='cuda'):
        super(HistogramLoss, self).__init__()
        self.bins = bins
        self.device = device
        
    def forward(self, source_tensor, target_tensor):
        N, C, H, W = source_tensor.shape
        loss = torch.tensor([], requires_grad=True, dtype=torch.float32).to(self.device)
        for i in range(N):
            for j in range(C):
                source = source_tensor[i, j, :, :].view(-1, 1)
                target = target_tensor[i, j, :, :].view(-1, 1)
                hist_source = torch.histc(source, bins=self.bins) / self.bins
                hist_target = torch.histc(target, bins=self.bins) / self.bins
                diff = torch.mean(torch.pow((hist_source - hist_target), 2))
                loss = loss + diff
        loss = loss / (C * N)
        loss = torch.mean(loss)
        return loss

class PolyDeconvLoss(torch.nn.Module):
    def __init__(self, alpha=6, beta=1, device='cuda'):
        super(HistogramLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
    def forward(self, source_tensor, target_tensor):
        a3 = alpha/2 - b + 2
        a2 = 3 * b - alpha - 6
        a1 = 5 - 3 * b + alpha / 2
        imout = a3 * img
        imout = filters.convolve2d(imout, kernel) + a2 * img
        imout = filters.convolve2d(imout, kernel) + a1 * img
        imout = filters.convolve2d(imout, kernel) + b * img
        # if the number of kernel matches the number of 
        if kernel.shape[1] == img.shape[1]:  # check how many color channels in the kernel
            return F.conv2d(img, kernel, groups=kernel.shape[1], padding=padding)
        else:
            img = [F.conv2d(img[:,c:c+1], kernel, padding=padding) for c in range(img.shape[1])]
            return torch.cat(img, dim=1) 
            N, C, H, W = source_tensor.shape


        loss = torch.tensor([], requires_grad=True, dtype=torch.float32).to(self.device)
        for i in range(N):
            for j in range(C):
                source = source_tensor[i, j, :, :].view(-1, 1)
                target = target_tensor[i, j, :, :].view(-1, 1)
                hist_source = torch.histc(source, bins=self.bins) / self.bins
                hist_target = torch.histc(target, bins=self.bins) / self.bins
                diff = torch.mean(torch.pow((hist_source - hist_target), 2))
                loss = loss + diff
        loss = loss / (C * N)
        loss = torch.mean(loss)
        return loss