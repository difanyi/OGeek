import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

class OGeekDataSet(data.Dataset):

    def __init__(self, path, type='train', transform=None, target_transform=None):

        self.path = path
        self.type = type

        self.transform = transform
        self.target_transform = target_transform

        dataset = np.load('./output/data_process.npz')

        self.x = dataset['X_{}'.format(type)]
        
        if type != 'test':
            self.target = dataset['y_{}'.format(type)]

    def __getitem__(self, index):

        query = self.x[index]
        # if self.transform is not None:
        #     print(self.transform)
        #     query = self.transform(query)

        if self.type!= 'test':
            target_ = self.target[index]

            # if self.target_transform is not None:
            #     target_ = self.target_transform(target_)

            return query, target_

        else: return query

    def __len__(self): 
        return len(self.x)

def make_ogeek_provider(path, batch_size):

    to_tensor = transforms.ToTensor()
    OGeek_train = OGeekDataSet(path, 'train', to_tensor, to_tensor)

    train_loader = data.DataLoader(OGeek_train, batch_size, shuffle = True)
    OGeek_val = OGeekDataSet(path, 'val', to_tensor, to_tensor)

    val_loader = data.DataLoader(OGeek_val, batch_size, shuffle =False)

    OGeek_test = OGeekDataSet(path, 'test', to_tensor, to_tensor)

    test_loader = data.DataLoader(OGeek_test, batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }





















