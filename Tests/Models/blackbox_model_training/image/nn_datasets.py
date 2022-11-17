from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PlainDataset(Dataset):
    def __init__(self, data, labels, transforms= None):
        super(PlainDataset, self).__init__()
        self.labels = labels
        self.data = data
        self.transforms = transforms
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input, label = self.data[index], self.labels[index]
        if not self.transforms is None:
            input = self.transforms(input)
        return input, label

######### Returns (0,1) normalized MNIST train (first 50,000 train) val (last 10,000 train) and test:
######### dataloader, dataset, arrays, labels 
def load_mnist(args, train_shuffle = True):
    tr_data = MNIST(root = "./Files/Data", download = True, train = True) 
    te_data  = MNIST(root = "./Files/Data", download = True, train = False)

    # tr_data.data = ((torch.reshape(tr_data.data, (len(tr_data.data), -1)) / 255) - 0.5) / 0.5 
    # te_data.data = ((torch.reshape(te_data.data, (len(te_data.data), -1)) / 255) - 0.5 ) / 0.5

    tr_data.data = torch.reshape(tr_data.data, (len(tr_data.data), -1)) / 255
    te_data.data = torch.reshape(te_data.data, (len(te_data.data), -1)) / 255

    tr_data_im   = tr_data.data[:-10000,:]
    tr_data_lab  = tr_data.targets[:-10000]
    
    val_data_im   = tr_data.data[-10000:]
    val_data_lab  = tr_data.targets[-10000:]

    te_data_im   = te_data.data 
    te_data_lab = te_data.targets


    # trans = torch.nn.Sequential(
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     )

    tr_data = PlainDataset(tr_data_im, tr_data_lab, transforms = None)
    val_data = PlainDataset(val_data_im, val_data_lab, transforms = None)
    te_data = PlainDataset(te_data_im, te_data_lab, transforms = None)



    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle = train_shuffle, drop_last=False) ##
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##
    te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##

    return tr_data_im, tr_data_lab, tr_data, tr_loader, val_data_im, val_data_lab, val_data, val_loader, te_data_im, te_data_lab, te_data, te_loader

######### Returns (0,1) normalized CIFAR10 train (first 40,000 train) val (last 10,000 train) and test:
######### dataloader, dataset, arrays, labels 
def load_cifar10(args, train_shuffle = True):
    print("HERE")
    tr_data = CIFAR10(root = "./", download = True, train = True) 
    te_data  = CIFAR10(root = "./", download = True, train = False)

    tr_data.data = torch.reshape(torch.tensor(tr_data.data), (len(tr_data.data), -1)) / 255
    te_data.data = torch.reshape(torch.tensor(te_data.data), (len(te_data.data), -1)) / 255

    tr_data_im   = tr_data.data[:-10000,:]
    tr_data_lab  = torch.tensor(tr_data.targets[:-10000])
    
    val_data_im   = tr_data.data[-10000:]
    val_data_lab  = torch.tensor(tr_data.targets[-10000:])

    te_data_im   = te_data.data 
    te_data_lab = te_data.targets

    tr_data = PlainDataset(tr_data_im, tr_data_lab, transforms = None)
    val_data = PlainDataset(val_data_im, val_data_lab, transforms = None)
    te_data = PlainDataset(te_data_im, te_data_lab, transforms = None)

    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle = train_shuffle, drop_last=False) ##
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##
    te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##

    return tr_data_im, tr_data_lab, tr_data, tr_loader, val_data_im, val_data_lab, val_data, val_loader, te_data_im, te_data_lab, te_data, te_loader

def load_fashionmnist(args, train_shuffle = True):
    print("HERE")
    tr_data = FashionMNIST(root = "./Files/Data", download = True, train = True) 
    te_data  = FashionMNIST(root = "./Files/Data", download = True, train = False)

    tr_data.data = torch.reshape(torch.tensor(tr_data.data), (len(tr_data.data), -1)) / 255
    te_data.data = torch.reshape(torch.tensor(te_data.data), (len(te_data.data), -1)) / 255

    tr_data_im   = tr_data.data[:-10000,:]
    tr_data_lab  = torch.tensor(tr_data.targets[:-10000])
    
    val_data_im   = tr_data.data[-10000:]
    val_data_lab  = torch.tensor(tr_data.targets[-10000:])

    te_data_im   = te_data.data 
    te_data_lab = te_data.targets

    tr_data = PlainDataset(tr_data_im, tr_data_lab, transforms = None)
    val_data = PlainDataset(val_data_im, val_data_lab, transforms = None)
    te_data = PlainDataset(te_data_im, te_data_lab, transforms = None)

    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle = train_shuffle, drop_last=False) ##
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##
    te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle = False, drop_last=False) ##

    return tr_data_im, tr_data_lab, tr_data, tr_loader, val_data_im, val_data_lab, val_data, val_loader, te_data_im, te_data_lab, te_data, te_loader