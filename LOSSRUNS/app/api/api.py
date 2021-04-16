"""API ROUTE
"""
import logging
import sys
from flask import Flask
from flask_restful import Api
from resources.controllers.ping_controller import PingController
from resources.controllers.invocations_controller import InvocationsController
from flask_cors import CORS

sys.path.append('/home/app/NPDB')

app = Flask(__name__)
cors = CORS(app)

logging.basicConfig(
    filename='api/logs/api.log',
    level=logging.DEBUG,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)

logging.getLogger('flask_cors').level = logging.DEBUG

api = Api(app)
api.add_resource(PingController, '/')
api.add_resource(InvocationsController, '/invocations')
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
