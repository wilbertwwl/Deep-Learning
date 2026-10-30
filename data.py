import torch, numpy as np, random
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms

SEED = 42

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_loaders(batch_size=128, root="./data"):
    set_seed()
    tf = transforms.ToTensor()  # piksel jadi rentang [0, 1]
    train_raw = datasets.FashionMNIST(root, train=True,  download=True, transform=tf)
    test_raw  = datasets.FashionMNIST(root, train=False, download=True, transform=tf)

    full = ConcatDataset([train_raw, test_raw])      # 70.000 citra
    n = len(full)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
