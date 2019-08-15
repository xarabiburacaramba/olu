from __future__ import print_function
from app import app
import json
import sys
import os
import numpy as np
from app.analytics.c import DB_Storage, OLU_Feature,  Imagee
from flask import request,  send_file
from io import BytesIO

np.set_printoptions(threshold=sys.maxsize)

#initialize db storage of OLU features
raster_file=os.environ['dem_raster_file']# add correct file_location
dbs=DB_Storage({'dbname':'%s' % os.environ['database'],'user':'%s' % os.environ['username'],'host':'%s' % os.environ['server'],'port':'%s' % os.environ['port'],'password':'%s'% os.environ['password']})
dbs.connect()
schema, table=os.environ['schema'], os.environ['table']

@app.route('/get_fields_geometry', methods=['GET'])
def get_params_of_get_fields_geometry():
    bbox = str(request.args.get('bbox'))
    return get_fields_geometry(bbox)
def get_fields_geometry(bbox):
    r=dbs.read_olu_features(schema,table,bbox)
    return json.dumps(r)
    
@app.route('/get_field_statistics', methods=['GET'])
def get_params_of_get_field_statistics():
    id = int(request.args.get('id'))
    onfly = str(request.args.get('onfly'))
    return get_field_statistics(id, onfly)
def get_field_statistics(id, onfly):
    if onfly=='false':
        r=dbs.read_field_statistics(schema,table,id)
        return json.dumps(r)
    elif onfly=='true':
        field=dbs.create_olu_feature(schema,table,id)
        field.add_raster_by_path(raster_file,'elevation')
        field.get_morphometric_characteristics()
        field.get_twi()
        return json.dumps(field.get_morphometric_statistics())
    else:
        return print('onfly parameter takes values true/false only .')

@app.route('/get_field_raster', methods=['GET'])
def get_params_of_get_field_raster():
    id = int(request.args.get('id'))
    kind = str(request.args.get('kind'))
    output = str(request.args.get('output'))
    onfly = str(request.args.get('onfly'))
    return get_field_raster(id, kind, output, onfly)
def get_field_raster(id,kind, output, onfly='false'):
    if onfly=='false':
        if  output=='image':
            image=dbs.read_raster(schema,table,id, kind)
            buffer = BytesIO()
            buffer.write(image)
            buffer.seek(0)
            return send_file(buffer,attachment_filename=("%s_raster.tif" % kind), as_attachment=True)
        elif output=='wms':
            wms=dbs.create_olu_feature(schema, table,id).generate_wms(kind)
            return wms
        else:
            return print('this output type is not supported .')
    elif onfly=='true':
        field=dbs.create_olu_feature(schema,table,id)
        field.add_raster_by_path(raster_file,'elevation')
        if kind=='twi':
            field.get_twi()
        else:
            field.get_morphometric_characteristics()
        buffer = BytesIO()
        buffer.write(field.read_raster(kind).export_into_memory())
        buffer.seek(0)
        return send_file(buffer,attachment_filename=("%s_raster.tif" % kind), as_attachment=True)
    else:
        return print('onfly parameter takes values true/false only .')
    
