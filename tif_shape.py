import os
import torch
import rasterio
import numpy as np
from torchvision import transforms

def main():
    file=os.path.join('/ste/rnd/User/yusuf/Istanbul_data',
                      'subset_6_of_S1A_IW_SLC__1SDV_20230916T160003_20230916T160030_050355_06100B_E5F6_split_Orb_Cal_deb_ML_TC_reprojected.tif')

    with rasterio.open(file) as dataset:
        print(type(dataset))
        print(f' number of bands: {dataset.count}')
        print(f'data height is {dataset.height} and width {dataset.width}')
        img_data=dataset.read()
        print(img_data.shape)
    #transpose shape to match format for ToTensor, i.e. (c,h,w) -> (h,w,c)

    #convert to tensor
    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.permute(1,2,0))])
    img_tensor=transform(img_data)

    print(f'shape of tensor is {img_tensor.shape}')

    #let's split vertical
    print('splitting vertically...')
    mp=img_tensor.shape[2]//2
    if (img_tensor.shape[2] % 2) != 0:
        x,y=torch.split(img_tensor,[mp,mp+1],dim=2)
    else:
        x,y = torch.split(img_tensor, mp, dim=2)
    #for i, el in enumerate(x):
    #    print(i, el.shape)
    print(x.shape, y.shape)

if __name__=="__main__":
    main()
