"""Invocations Module
"""
import sys
sys.path.append('/home/app/NPDB')
import tempfile
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage
from main_npdb_cover_beta import run
import hashlib
import time

def file_name():
    ts = str(time.time())
    return hashlib.md5(ts.encode('utf-8')).hexdigest()


class InvocationsController(Resource):
    """InvocationsController Class
    """
    def post(self):
        """Upload File
            Returns:
                DICT: file name with md5 hash
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            file = self.__pdf_file(tmpdirname)
            tmp_file = "%s/%s.pdf" % (tmpdirname, file_name())
            file.save(tmp_file)
            return {"data": run(tmp_file)}

    @staticmethod
    def __pdf_file(tmpdirname):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=FileStorage, location='files')
        args = parse.parse_args()
        return args['file']
