from osgeo import gdal
import csv

raster_dictionary={'B02':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B02.tif',
'B03':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B03.tif',
'B04':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B04.tif',
'B08':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B08.tif'}


b02_s=gdal.Open(raster_dictionary['B02'])
b02_band=b02_s.GetRasterBand(1)

b03_s=gdal.Open(raster_dictionary['B03'])
b03_band=b03_s.GetRasterBand(1)

b04_s=gdal.Open(raster_dictionary['B04'])
b04_band=b04_s.GetRasterBand(1)

b08_s=gdal.Open(raster_dictionary['B08'])
b08_band=b08_s.GetRasterBand(1)


def get_offset_by_coordinates(raster,x,y):
 geotransform = raster.GetGeoTransform()
 originX = geotransform[0]
 originY = geotransform[3]
 pixelWidth = geotransform[1]
 pixelHeight = geotransform[5]
 xOffset = int((x - originX)/pixelWidth)
 yOffset = int((y - originY)/pixelHeight)
 return xOffset,yOffset
 
def get_value_in_the_coordinate(raster,x,y):
 return raster.ReadAsArray(xoff=x,yoff=y,win_xsize=1,win_ysize=1)[0][0]
 
def get_value(dataset,band,x,y):
 '''Get altitude of the point.
  Parameters: x - longitude, y - latitude. Return value: altitude above sea level in meters.
  Source of data SRTM 1 arcsecond by USGS'''
 offset=get_offset_by_coordinates(dataset,x,y)
 value=get_value_in_the_coordinate(band,offset[0],offset[1])
 return int(value)


columns=['x','y','ID','B2','B3','B4','B8']

with open("/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/prediction_grid.csv", 'a') as f:
 writer = csv.writer(f)
 writer.writerow(columns)
  
x=1862658
y=6197051
id=1

while x<=1866969:
 print(x)
 while y<=6199952:
  print(y)
  b02_value=get_value(b02_s,b02_band,x,y)
  print(b02_value)
  b03_value=get_value(b03_s,b03_band,x,y)
  b04_value=get_value(b04_s,b04_band,x,y)
  b08_value=get_value(b08_s,b08_band,x,y)
  row=[x,y,id,int(b02_value),int(b03_value),int(b04_value),int(b08_value)]
  with open("/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/prediction_grid.csv", 'a') as f:
   writer = csv.writer(f)
   writer.writerow(row)
  id+=1
  y+=10
 x+=10
 y=6197051