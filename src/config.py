import os
basedir = os.path.abspath(os.path.dirname(__file__))
class Config(object):
 SECRET_KEY = os.environ.get('SECRET_KEY') or 'fdskljfbhke384638674twaibdrgf2GHASUIYGD2698^*&^'
 DATA_STATIC_PATH = 'static/data/'
 USER_STATIC_PATH = 'static/user_name/data/'
 SESSION_TYPE = 'filesystem'
 SESSION_FILE_DIR = '%s/flask_session' % basedir
 FLASK_DEBUG = 1
