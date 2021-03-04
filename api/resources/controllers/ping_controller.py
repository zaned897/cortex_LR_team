"""PingController Module
"""
from flask_restful import Resource

class PingController(Resource):

    """PingController Class
    """

    def get(self):
        """Get Ping
            Returns:
                TUPLE: Empty response body and status code 200
        """
        return {}, 200
