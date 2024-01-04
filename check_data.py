import argparse
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def main(args):
    in_pth=args.input_file
    out_pth=args.output_file

    tiff_data=gdal.Open(in_pth)

    print(f'total number of channels is: {tiff_data.RasterCount}')

    for bnd_num in range(1,tiff_data.RasterCount+1):
        bnd=tiff_data.GetRasterBand(bnd_num)
        description=bnd.GetDescription()
        if description:
            print(f'band description of band number {bnd_num} is: {description}')

    #data=tiff_data.ReadAsArray()

    subset_data=tiff_data.GetRasterBand(1).ReadAsArray()
    #close to free up some resources
    tiff_data=None

    #apply transformations
    data=np.log1p(subset_data)

    threshold=2.5*np.mean(subset_data)
    subset_data=np.clip(subset_data,0,threshold)

    #scale
    subset_data*=255/np.max(subset_data)

    #plot
    plt.imshow(subset_data,cmap='gray')
    plt.savefig(out_pth)


if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("-f", "--input_file", type=str, required=True, help='path to tif file')

    parser.add_argument("-o", "--output_file", type=str, required=True, help='path to tif file')

    args=parser.parse_args()

    main(args)
