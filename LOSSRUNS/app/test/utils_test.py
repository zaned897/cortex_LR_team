import json
import os
from contextlib import contextmanager

class Utils(object):
    def __init__(self, arg):
        super(Utils, self).__init__()
        self.expected_data_path = None

    def replace_ext(self, file):
        return self.expected_data_path + file.name.replace('.pdf', '.json')

    @staticmethod
    def pdf_file_path(file):
        return "/home/app/%s" % file.path

    @staticmethod
    @contextmanager
    def get_files():
      try:
          print('Searching for pdf files...')
          with os.scandir('test/resources/pdf_samples') as entries:
            yield entries
      finally:
        print('All pdf files were checked.')

    @contextmanager
    def open_json(self, path):
        file_stream = open(path, "r")
        try:
            data = file_stream.read()
            yield json.loads(data)
        finally:
            file_stream.close()