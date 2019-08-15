import gdal
import gdalnumeric
import ogr
import osr
#import gdal_array
from PIL import Image, ImageDraw
import os
import numpy as np

def get_olu_feature_by_id(id):
 '''
 Function to get wkt geometry of selected of Open Land Use database.
 Hardcoded variables that need to be replaced are host,db,user,pw,lyr_name. 
 They are needed to establish connection directly with PostgreSQL database,
 where Open Land Use database is stored.
 The only parameter of the function is id refering to id of Open Land Use feature,
 that will be fetched.
 '''
 host='127.0.0.1'
 db='postgis_24_sample'
 user='postgres'
 pw='postgres12!'
 lyr_name='european_land_use.areas_master'
 connString = "PG: host=%s dbname=%s user=%s password=%s" % (host,db,user,pw)
 conn = ogr.Open(connString)
 lyr = conn.GetLayer( lyr_name )
 feature=lyr.GetFeature(id)
 geometry=feature.GetGeometryRef()
 source = osr.SpatialReference()
 source.ImportFromEPSG(3857)
 target = osr.SpatialReference()
 target.ImportFromEPSG(4326)
 transform = osr.CoordinateTransformation(source, target)
 geometry.Transform(transform)
 geometry_wkt=geometry.ExportToWkt()
 feature=None
 lyr=None
 conn=None
 return geometry_wkt
 
 
def world_to_pixel(geo_matrix, x, y):
 '''
 Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
 the pixel location of a geospatial coordinate; from:
 http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#clip-a-geotiff-with-shapefile
 '''
 ulX = geo_matrix[0]
 ulY = geo_matrix[3]
 xDist = geo_matrix[1]
 yDist = geo_matrix[5]
 rtnX = geo_matrix[2]
 rtnY = geo_matrix[4]
 pixel = int((x - ulX) / xDist)
 line = int((ulY - y) / xDist)
 return (pixel, line)

 
def array_to_image(a):
 '''
 Converts a gdalnumeric array to a Python Imaging Library (PIL) Image.
 '''
 i = Image.fromstring('L',(a.shape[1], a.shape[0]),
 (a.astype('b')).tostring())
 return i

def image_to_array(i):
 '''
 Converts a Python Imaging Library (PIL) array to a gdalnumeric image.
 '''
 a = gdalnumeric.fromstring(i.tobytes(), 'b')
 a.shape = i.im.size[1], i.im.size[0]
 return a


 
class Imagee:
 '''
 Data is a double(x,y)-array image.
 Metadata is a dictionary object. One of the metadata keys should be 'affine_transformation'.
 It holds affine transformation parameters from ogr.gdal.GetGeoTransform() function.
 Typically represented by gdal array data type.
 Another recommended metadata key in the dictionary is 'nodata' key referring to which value should be neglected.
 '''
 def __init__ (self, dataarray=None, metadata=None):
  '''
  Initialize the Imagee object.
  It is needed to provide numpy array (values in 2D space) as well as metadata,
  where 'affine_transformation' and 'nodata' keys are important.
  '''
  self._data = dataarray
  self._metadata = metadata
  
 def get_metadata(self):
  '''
  Returns metadata dictionary.
  '''
  return self._metadata
  
 def get_data(self):
  '''
  Returns 2D matrix of values.
  '''
  return self._data
  
  
 def export_as_tif(self,filename):
  '''
  Export self data as GeoTiff 1-band image. 
  Output filename should be provided as a parameter.
  '''
  nrows,ncols=self._data.shape
  geotransform = self._metadata['affine_transformation']
  output_raster = gdal.GetDriverByName('GTiff').Create(filename, ncols, nrows, 1, gdal.GDT_Float32)
  output_raster.SetGeoTransform(geotransform)
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(4326)
  #srs.ImportFromWkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')
  output_raster.SetProjection(srs.ExportToWkt())
  output_raster.GetRasterBand(1).WriteArray(self._data)
  output_raster.GetRasterBand(1).SetNoDataValue(-32767)
  output_raster.FlushCache()
  del output_raster
  
 def clip_by_olu_shape(self, olu_id, nodata=-32767):
  '''
  Clip an Imagee by Open Land Use feature.
  '''
  rast = self._data
  gt=self._metadata['affine_transformation']
  
  # Get the first feature
  geom_wkt = get_olu_feature_by_id(olu_id)
  poly=ogr.CreateGeometryFromWkt(geom_wkt)

  # Convert the layer extent to image pixel coordinates
  minX, maxX, minY, maxY = poly.GetEnvelope()
  ulX, ulY = world_to_pixel(gt, minX, maxY)
  lrX, lrY = world_to_pixel(gt, maxX, minY)

  # Calculate the pixel size of the new image
  pxWidth = int(lrX - ulX)
  pxHeight = int(lrY - ulY)

  # If the clipping features extend out-of-bounds and ABOVE the raster...
  if gt[3] < maxY:
   # In such a case... ulY ends up being negative--can't have that!
   iY = ulY
   ulY = 0

  clip = rast[ulY:lrY, ulX:lrX]

  # Create a new geomatrix for the image
  gt2 = list(gt)
  gt2[0] = minX
  gt2[3] = maxY

  # Map points to pixels for drawing the boundary on a blank 8-bit,
  #   black and white, mask image.
 
  raster_poly = Image.new('L', (pxWidth, pxHeight), 1)
  rasterize = ImageDraw.Draw(raster_poly)
  
  def rec(poly_geom):
   '''
   Recursive drawing of parts of multipolygons over initialized PIL Image object using ImageDraw.Draw method.
   '''
   if poly_geom.GetGeometryCount()==0:
    points=[]
    pixels=[]
    for p in range(poly_geom.GetPointCount()):
     points.append((poly_geom.GetX(p), poly_geom.GetY(p)))
    for p in points:
     pixels.append(world_to_pixel(gt2, p[0], p[1]))
    rasterize.polygon(pixels, 0)
   if poly_geom.GetGeometryCount()>=1:
    for j in range(poly_geom.GetGeometryCount()):
     rec(poly_geom.GetGeometryRef(j))
	 
  rec(poly)

  mask = image_to_array(raster_poly)

  # Clip the image using the mask
  try:
   clip = gdalnumeric.choose(mask, (clip, nodata))

  # If the clipping features extend out-of-bounds and BELOW the raster...
  except ValueError:
   # We have to cut the clipping features to the raster!
   rshp = list(mask.shape)
   if mask.shape[-2] != clip.shape[-2]:
    rshp[0] = clip.shape[-2]

   if mask.shape[-1] != clip.shape[-1]:
    rshp[1] = clip.shape[-1]

   mask.resize(*rshp, refcheck=False)

   clip = gdalnumeric.choose(mask, (clip, nodata))
  
  #self._data=clip
  #self._metadata['affine_transformation'],self._metadata['ul_x'],self._metadata['ul_y']=gt2,ulX,ulY
  d={}
  d['affine_transformation'],d['ul_x'],d['ul_y'],d['nodata']=gt2,ulX,ulY,-32767  
  return (clip, d)
  
 def clip_by_olu_shape_bb_buffer(self, olu_id, buffer=0):
 
  rast = self._data
  gt=self._metadata['affine_transformation']
  
  # Get the first feature
  geom_wkt = get_olu_feature_by_id(olu_id)
  poly=ogr.CreateGeometryFromWkt(geom_wkt)

  # Convert the layer extent to image pixel coordinates
  minX, maxX, minY, maxY = poly.GetEnvelope()
  minX-=(buffer*gt[1]+gt[1])
  maxX+=(buffer*gt[1]+gt[1])
  minY+=(buffer*gt[5]+gt[5])
  maxY-=(buffer*gt[5]+gt[5])
  ulX, ulY = world_to_pixel(gt, minX, maxY)
  lrX, lrY = world_to_pixel(gt, maxX, minY)
  

  # Calculate the pixel size of the new image
  pxWidth = int(lrX - ulX)
  pxHeight = int(lrY - ulY)

  clip = rast[ulY:lrY, ulX:lrX]

  # Create a new geomatrix for the image
  gt2 = list(gt)
  gt2[0] = minX
  gt2[3] = maxY
  
  d={}
  d['affine_transformation'],d['ul_x'],d['ul_y'],d['nodata']=gt2,ulX,ulY,-32767
  
  return (clip, d)
  
 def calculate_slope(self):
  '''
  Calculate slope from self data of DEM image.
  '''
  x, y = np.gradient(self._data)
  slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
  return (slope,self._metadata)
  
 def calculate_azimuth(self):
  '''
  Calculate azimuth from self data of DEM image.
  '''
  x, y = np.gradient(self._data)
  aspect = (np.arctan2(-x, y))*180/np.pi
  return (aspect,self._metadata)
  
 def get_min_value(self):
  '''
  Get self min value excluding self nodata value.
  '''
  return np.min(self._data[np.where(self._data!=self._metadata['nodata'])])
  
 def get_max_value(self):
  '''
  Get self max value excluding self nodata value.
  '''
  return np.max(self._data[np.where(self._data!=self._metadata['nodata'])])
 
 def get_mean_value(self):
  '''
  Get self mean value excluding self nodata values.
  '''
  return np.mean(self._data[np.where(self._data!=self._metadata['nodata'])])
  
 def get_median_value(self):
  '''
  Get self median value excluding self nodata values.
  '''
  return np.median(self._data[np.where(self._data!=self._metadata['nodata'])])
  
def main(olu_id, raster_file_name = 'G://GIS/data/dem_weinviertel.tif'):
 '''
 Main function calculating all zonal statistics such as min/max/median/mean slope and aspect of give Open Land Use feature
 and outputting them as a dictionary. 
 Needed input parameters are id of Open Land Use feature and also file name of DEM raster.
 '''
 ds = gdal.Open(raster_file_name)
 metadata_dict={}
 metadata_dict['affine_transformation']=ds.GetGeoTransform()
 dem=Imagee(np.array(ds.GetRasterBand(1).ReadAsArray()),metadata_dict)
 dem_clipped_by_bounding_box=Imagee(*dem.clip_by_olu_shape_bb_buffer(olu_id,1))
 dem_clipped_by_bounding_box_slope=Imagee(*dem_clipped_by_bounding_box.calculate_slope())
 dem_clipped_by_bounding_box_azimuth=Imagee(*dem_clipped_by_bounding_box.calculate_azimuth())
 dem=Imagee(*dem_clipped_by_bounding_box.clip_by_olu_shape(olu_id))
 dem_slope=Imagee(*dem_clipped_by_bounding_box_slope.clip_by_olu_shape(olu_id))
 dem_azimuth=Imagee(*dem_clipped_by_bounding_box_azimuth.clip_by_olu_shape(olu_id))
 #dem.export_as_tif('G://GIS/data/'+str(olu_id)+'_dem.tif')
 #dem_slope.export_as_tif('G://GIS/data/'+str(olu_id)+'_dem_slope.tif')
 #dem_azimuth.export_as_tif('G://GIS/data/'+str(olu_id)+'_dem_azimuth.tif')
 zonal_statistics={}
 zonal_statistics['min_elevation'],zonal_statistics['max_elevation'],zonal_statistics['mean_elevation'],zonal_statistics['median_elevation']=dem.get_min_value(),dem.get_max_value(),dem.get_mean_value(),dem.get_median_value()
 zonal_statistics['min_slope'],zonal_statistics['max_slope'],zonal_statistics['mean_slope'],zonal_statistics['median_slope']=dem_slope.get_min_value(),dem_slope.get_max_value(),dem_slope.get_mean_value(),dem_slope.get_median_value()
 zonal_statistics['min_azimuth'],zonal_statistics['max_azimuth'],zonal_statistics['mean_azimuth'],zonal_statistics['median_azimuth']=dem_azimuth.get_min_value(),dem_azimuth.get_max_value(),dem_azimuth.get_mean_value(),dem_azimuth.get_median_value()
 return zonal_statistics