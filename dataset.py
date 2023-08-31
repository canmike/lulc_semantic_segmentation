
import torch
from torch.utils import data
from utils import get_img

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 indexes,
                 file_path='/content/data/C2Seg_AB/train',
                 transform=None,
                 use_sar=True,
                 use_msi=True,
                 use_hsi=True,
                 reduce_hsi=True,
                 reduce_bands=20
                 ):
        self.file_path = file_path
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.transform = transform
        self.indexes = indexes
        self.use_sar = use_sar
        self.use_msi = use_msi
        self.use_hsi = use_hsi
        self.reduce_hsi=reduce_hsi
        self.reduce_bands=20


    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,
                    index: int):
        data_index = self.indexes[index]

        X, y = get_img(data_index, self.file_path, use_hsi=self.use_hsi, reduce_hsi=self.reduce_hsi, reduce_bands=self.reduce_bands, use_sar=self.use_sar, use_msi=self.use_msi)
        X = torch.squeeze(X)
        # Preprocessing
        if self.transform is not None:
            X, y = self.transform(X, y)

        # Typecasting
        X, y = X.type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return X, y
