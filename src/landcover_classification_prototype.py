import psycopg2
import os
import csv
import numpy as np
from osgeo import gdal
from app.analytics.c import DB_Storage, OLU_Feature, Imagee

schema='european_land_use'
table='areas_master'
	
raster_dictionary={'B02':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B02.tif',
'B03':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B03.tif',
'B04':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B04.tif',
'B08':'/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/B08.tif'}


b02_s=gdal.Open(raster_dictionary['B02'])
metadata_dict={}
metadata_dict['affine_transformation']=b02_s.GetGeoTransform()
b02=Imagee(np.array(b02_s.GetRasterBand(1).ReadAsArray()),metadata_dict)

b03_s=gdal.Open(raster_dictionary['B03'])
metadata_dict={}
metadata_dict['affine_transformation']=b03_s.GetGeoTransform()
b03=Imagee(np.array(b03_s.GetRasterBand(1).ReadAsArray()),metadata_dict)

b04_s=gdal.Open(raster_dictionary['B04'])
metadata_dict={}
metadata_dict['affine_transformation']=b04_s.GetGeoTransform()
b04=Imagee(np.array(b04_s.GetRasterBand(1).ReadAsArray()),metadata_dict)

b08_s=gdal.Open(raster_dictionary['B08'])
metadata_dict={}
metadata_dict['affine_transformation']=b08_s.GetGeoTransform()
b08=Imagee(np.array(b08_s.GetRasterBand(1).ReadAsArray()),metadata_dict)

columns=['x','y','B2','B3','B4','B8','Class','Class_ID']

with open("/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/train_labels.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
with open("/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/test_labels.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(columns)


dbs=DB_Storage({'dbname':'%s' % os.environ['database'],'user':'%s' % os.environ['username'],'host':'%s' % os.environ['server'],'port':'%s' % os.environ['port'],'password':'%s'% os.environ['password']})
dbs.connect()

connection=psycopg2.connect("dbname=gis user=postgres host=localhost port=5432 password=postgres")
cursor=connection.cursor()
values=[]
cursor.execute('select distinct(hilucs_value) from european_land_use.areas_master where st_intersects(geom,st_setsrid(st_makebox2d(st_makepoint(1821186,6160257), st_makepoint(1911021,6231534)),3857))')
for row in cursor:
	values.append(int(row[0]))

for value in values:
	ids=[]
	cursor.execute(('select * from european_land_use.areas_master where st_intersects(geom,st_setsrid(st_makebox2d(st_makepoint(1821186,6160257), st_makepoint(1911021,6231534)),3857)) and hilucs_value=%s limit 800') % value)
	for row in cursor:
		ids.append(row[0])
	len_ids=len(ids)
	train_length=int(len_ids*0.7)
	test_length=len_ids-train_length
	for id in ids[0:train_length]:
		try:
			feature1=dbs.create_olu_feature(schema,table,id)
			b02_cropped=Imagee(*b02.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b02_cropped,'B02')
			b03_cropped=Imagee(*b03.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b03_cropped,'B03')
			b04_cropped=Imagee(*b04.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b04_cropped,'B04')
			b08_cropped=Imagee(*b08.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b08_cropped,'B08')
			xsize,ysize=feature1.read_raster('B02').get_data().shape
			for i in range(10):
				r_num,c_num=np.random.randint(xsize),np.random.randint(ysize)
				b02_value=feature1.read_raster('B02').get_data()[r_num,c_num]
				if np.ma.is_masked(b02_value):
					while np.ma.is_masked(b02_value)==True:
						r_num,c_num=np.random.randint(xsize),np.random.randint(ysize)
						b02_value=feature1.read_raster('B02').get_data()[r_num,c_num]
				b03_value=feature1.read_raster('B03').get_data()[r_num,c_num]
				b04_value=feature1.read_raster('B04').get_data()[r_num,c_num]
				b08_value=feature1.read_raster('B08').get_data()[r_num,c_num]
				x,y=feature1.read_raster('B02').image_to_geo_coordinates(r_num,c_num)
				row=[x,y,int(b02_value),int(b03_value),int(b04_value),int(b08_value),int(value) if len_ids>=100 else 600,(values.index(value)+1) if len_ids>=100 else 99]
				with open("/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/train_labels.csv", 'a') as f:
					writer = csv.writer(f)
					writer.writerow(row)
		except:
			pass
                
	for id in ids[train_length:]:
		try:
			feature1=dbs.create_olu_feature(schema,table,id)
			b02_cropped=Imagee(*b02.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b02_cropped,'B02')
			b03_cropped=Imagee(*b03.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b03_cropped,'B03')
			b04_cropped=Imagee(*b04.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b04_cropped,'B04')
			b08_cropped=Imagee(*b08.clip_by_shape(feature1.get_geomwkt(),-32767))
			feature1.add_raster(b08_cropped,'B08')
			xsize,ysize=feature1.read_raster('B02').get_data().shape
			for i in range(10):
				r_num,c_num=np.random.randint(xsize),np.random.randint(ysize)
				b02_value=feature1.read_raster('B02').get_data()[r_num,c_num]
				if np.ma.is_masked(b02_value):
					while np.ma.is_masked(b02_value)==True:
						r_num,c_num=np.random.randint(xsize),np.random.randint(ysize)
						b02_value=feature1.read_raster('B02').get_data()[r_num,c_num]
				b03_value=feature1.read_raster('B03').get_data()[r_num,c_num]
				b04_value=feature1.read_raster('B04').get_data()[r_num,c_num]
				b08_value=feature1.read_raster('B08').get_data()[r_num,c_num]
				x,y=feature1.read_raster('B02').image_to_geo_coordinates(r_num,c_num)
				row=[x,y,int(b02_value),int(b03_value),int(b04_value),int(b08_value),int(value) if len_ids>=100 else 600,(values.index(value)+1) if len_ids>=100 else 99]
				with open("/home/dima/Documents/data/R_randomForest/S2B_MSIL1C_20180819T100019_N0206_R122_T33UXP_20180819T141300.SAFE/GRANULE/L1C_T33UXP_A007584_20180819T100323/IMG_DATA/test_labels.csv", 'a') as f:
					writer = csv.writer(f)
					writer.writerow(row)
		except:
			pass

#read into memory bands 2,3,4,8:
#for category in categories:
#	select objects in database counted from 0 to 100 that are located on photo:
#		select 10 random pixels in one of those 100 objects
#			for band in bands:
#				see what values it has in the band:
#					add to the csv file
#16.36 48.32
#17.167 48.744
#gdalwarp -of "GTiff" -co "TFW=Yes" -co "TILED=YES" -tr 10 10 -s_srs EPSG:32633 -t_srs EPSG:3857 -te 1821186 6160257 1911021 6231534  T33UXP_20180819T100019_B08.jp2 B08.tif
