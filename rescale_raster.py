import argparse
from osgeo import gdal

def main(args):
    in_pth = args.input_file
    scale = args.scale
    out_pth = args.output_file
    width=args.width
    height=args.height

    tiff_data=gdal.Open(in_pth)

    #check raster info
    print(f'projection is {tiff_data.GetProjection}')
    print(f'num cols {tiff_data.RasterXSize}')
    print(f'projection is {tiff_data.RasterYSize}')

    #re-scale raster info
    if scale:
        scaled_tiff=gdal.Translate(out_pth,tiff_data,width=width, height=height)
        print('data re-scaled and saved!')



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--input_file", type=str, required=True, help='path to tif file')

    parser.add_argument("-s", "--scale", action="store_true", help='only scale if true')

    parser.add_argument("-w", "--width", type=int, help='scaled width of tif file')
    parser.add_argument("-he", "--height", type=int,  help='scaled height of tif file')
    parser.add_argument("-o", "--output_file", type=str, help='path to tif file')

    args = parser.parse_args()

    main(args)