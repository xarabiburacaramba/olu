import gdal
import gdalnumeric
import ogr
import osr
from PIL import Image, ImageDraw
import os
import numpy as np
from numpy import ma
import math
import pygeoprocessing
from pygeoprocessing import routing
import psycopg2
import psycopg2.extras


class DB_Storage():
    def __init__(self, connection_parameters,  type='postgresql'):
        self._connection_parameters=connection_parameters
        self._type=type
    def connect(self):
        self._connection = psycopg2.connect("dbname='%s' user='%s' host='%s' port=%s password='%s'" % (self._connection_parameters['dbname'],self._connection_parameters['user'] , self._connection_parameters['host'], self._connection_parameters['port'], self._connection_parameters['password']) )
    def disconnect(self):
        self._connection.close()
    def update(self, update_statement):
        cursor=self._connection.cursor()
        cursor.execute(update_statement)
        self._connection.commit()
        cursor.close()
    def alter(self, alter_statement):
        cursor=self._connection.cursor()
        cursor.execute(alter_statement)
        self._connection.commit()
        cursor.close()
    def update_olu_features(self, schema,  table):
        cursor=self._connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("select id, municipal_code as nuts_id, st_astext(geom) as geom_wkt from %s.%s where st_npoints(geom)>2" %(schema, table) )
        for i in cursor:
            yield (OLU_Feature(**i).update(self, schema,  table,  type='add_rasters'))
        cursor.close()


def image_to_array(i):
	'''
	Converts a Python Imaging Library (PIL) array to a gdalnumeric image.
	'''
	a = gdalnumeric.fromstring(i.tobytes(), 'b')
	a.shape = i.im.size[1], i.im.size[0]
	return a



class OLU_Feature():
    '''Class mirror to db. Either using SQL-Alchemy, reading class from outside through Restfull API.
    '''
    
    def __init__ (self, id, nuts_id,  geom_wkt,  user_id=0):
        '''
            inicializace tridy
        '''
        self._id = id
        self._nuts_id=nuts_id
        self._geom_wkt = geom_wkt
        self._user_id = user_id
        self._rasters=[]
        self._rasters_index=[]
        
    def get_id(self):
        return self._id
    
    def get_userid(self):
        return self._user_id
    
    def get_geomwkt(self):
        return self._geom_wkt
    
    def get_nutsid(self):
        return self._nuts_id
    
    def add_raster(self,raster,kind):
        if 'kind' in self._rasters_index:
            raster_index=self._rasters_index.index(kind)
            self._rasters[raster_index]=raster
        else:
            self._rasters.append(raster)
            self._rasters_index.append(kind)
        
    def read_raster(self,kind):
        raster_index=self._rasters_index.index(kind)
        try:
            print(kind)
            return self._rasters[raster_index]
        except:
            print('Error!')
            
    def read_dem(self,raster_path):
        ds = gdal.Open(raster_path)
        metadata_dict={}
        metadata_dict['affine_transformation']=ds.GetGeoTransform()
        dem=Imagee(np.array(ds.GetRasterBand(1).ReadAsArray()),metadata_dict)
        dem_cropped=Imagee(*dem.clip_by_shape(self._geometry_wkt,-32767))
        self.add_raster(dem_cropped,'dem')
        print('Added!')
        
    def get_morphometric_characteristics(self, return_images=True):
        dem=self.read_raster('dem')
        dem_slope=dem.calculate_slope()
        dem_azimuth=dem.calculate_azimuth()
        self.add_raster(Imagee(*dem_slope),'slope')
        self.add_raster(Imagee(*dem_azimuth),'azimuth')
        if return_images:
            return((self.read_raster('slope'),self.read_raster('azimuth')))
    
    def get_twi(self):
        dem_export_fn='/tmp/dem_%s.tif' % self._id
        cdem_export_fn='/tmp/cdem_%s.tif' % self._id
        direction_export_fn='/tmp/direction_%s.tif' % self._id
        accumulation_export_fn='/tmp/accumulation_%s.tif' % self._id
        dem=self.read_raster('dem')
        dem.export_as_tif(dem_export_fn)
        routing.fill_pits((dem_export_fn,1),cdem_export_fn)
        routing.flow_dir_mfd((cdem_export_fn,1),direction_export_fn)
        routing.flow_accumulation_mfd((direction_export_fn,1),accumulation_export_fn)
        ds = gdal.Open(accumulation_export_fn)
        metadata_dict={}
        metadata_dict['affine_transformation']=ds.GetGeoTransform()
        twi=Imagee(np.array(ds.GetRasterBand(1).ReadAsArray()),metadata_dict)
        self.add_raster(twi,'twi')
        return twi
    
    def generate_contour_lines(self, interval):
        dem=self.read_raster('dem')
        contour_lines=dem.generate_contour_lines(interval)
        return contour_lines
        
    def update(self, db_storage, schema,  table,  type):
        update_statement=('update %s.%s set dem = ST_SetValues(ST_AddBand(ST_MakeEmptyRaster(%s, %s, %s, %s, 1, 1, 0, 0, 4326),\'32BF\'::text, 2, %s), 1, 1, 1,ARRAY%s::double precision[][])) where id=%s;' %(schema, table,self.read_raster('dem').get_data().shape[1],self.read_raster('dem').get_data().shape[0],self.read_raster('dem').get_metadata()['affine_transformation'][0],self.read_raster('dem').get_metadata()['affine_transformation'][3],self.read_raster('dem').get_metadata()['nodata'],(np.array2string(self.read_raster('dem').get_data().filled(self._value_matrix_of_pixels.get_metadata()['nodata']), separator=',',formatter={'float_kind':lambda x: "%.4f" % x},prefix='[',suffix=']').replace('\n', '')),  self._id) )
        db_storage.update(update_statement)


class Imagee():
	'''Class to read and work with raster data - DEM, Sentinel Imagery
	'''
	def __init__ (self, dataarray=None, metadata=None):
		'''
		Initialize the Imagee object.
		It is needed to provide numpy array (values in 2D space) as well as metadata,
		where 'affine_transformation' and 'nodata' keys are important.
		'''
		self._metadata = metadata
		if type(dataarray)==ma.core.MaskedArray:
			dataarray = dataarray.filled(-32767)
			self._data = ma.array(dataarray,mask=[dataarray==-32767])
		elif 'nodata' in self._metadata:
			self._data = ma.array(dataarray,mask=[dataarray==self._metadata['nodata']])
		else:
			self._data = ma.array(dataarray,mask=[dataarray==-32767])
			
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
	
	def clip_by_shape(self, geom_wkt, nodata=-32767):
		'''
		Clip an Imagee by vector feature.
		'''
		rast = self._data
		gt=self._metadata['affine_transformation']
		poly=ogr.CreateGeometryFromWkt(geom_wkt)
		# Convert the layer extent to image pixel coordinates
		minX, maxX, minY, maxY = poly.GetEnvelope()
		#ulX, ulY = world_to_pixel(gt, minX, maxY)
		#lrX, lrY = world_to_pixel(gt, maxX, minY)
		ulX, ulY = math.floor((minX-gt[0])/gt[1]),math.floor((maxY-gt[3])/gt[5])
		lrX,lrY = math.ceil((maxX-gt[0])/gt[1]),math.ceil((minY-gt[3])/gt[5])
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
		#gt2[0] = minX
		#gt2[3] = maxY
		gt2[0] = ulX*gt[1]+gt[0]
		gt2[3] = ulY*gt[5]+gt[3]
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
					#pixels.append(world_to_pixel(gt2, p[0], p[1]))
					pixels.append((int((p[0]-gt2[0])/gt2[1]),int((p[1]-gt2[3])/gt2[5])))
				rasterize.polygon(pixels, 0)
			if poly_geom.GetGeometryCount()>=1:
				for j in range(poly_geom.GetGeometryCount()):
					rec(poly_geom.GetGeometryRef(j))
			
		rec(poly)
		mask = image_to_array(raster_poly)
		# Clip the image using the mask
		try:
			#clip = gdalnumeric.choose(mask, (clip, nodata))
			clip=ma.array(clip,mask=mask)
			clip=clip.astype(float).filled(fill_value=nodata)
		# If the clipping features extend out-of-bounds and BELOW the raster...
		except ValueError:
			# We have to cut the clipping features to the raster!
			rshp = list(mask.shape)
			if mask.shape[-2] != clip.shape[-2]:
				rshp[0] = clip.shape[-2]
			if mask.shape[-1] != clip.shape[-1]:
				rshp[1] = clip.shape[-1]
			mask.resize(*rshp, refcheck=False)
			#clip = gdalnumeric.choose(mask, (clip, nodata))
			clip=ma.array(clip,mask=mask)
			clip=clip.astype(float).filled(fill_value=nodata)
		#self._data=clip
		#self._metadata['affine_transformation'],self._metadata['ul_x'],self._metadata['ul_y']=gt2,ulX,ulY
		d={}
		d['affine_transformation'],d['ul_x'],d['ul_y'],d['nodata']=gt2,ulX,ulY,nodata
		#clip = ma.array(clip,mask=[clip==nodata])
		return (clip, d)
	
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
		output_raster.GetRasterBand(1).SetNoDataValue(self._metadata['nodata'])
		output_raster.FlushCache()
		del output_raster

def main():
    db=DB_Storage({'dbname':'gis','user':'postgres','host':'localhost','port':'5432','password':'postgres'})
    db.connect()
    db.alter('alter table european_land_use.areas_master add column dem raster;')
    db.update_olu_features('european_land_use', 'areas_master', type='morphometric')
    # follows method that dumps database and sends it to ATOS cloud or just dumps and stores temporary on server for others to be able to download dump file

if __name__ == '__main__':
    main()
    

