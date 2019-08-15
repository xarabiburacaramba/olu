from flask import Flask
from config import Config
#from flask_login import LoginManager

from flask_session import Session
from werkzeug.contrib.cache import SimpleCache

app = Flask(__name__)
app.config.from_object(Config)

# think how to use user sessions
Session(app)

# using cache
cache = SimpleCache()

app.app_context().push()

from app import routes
