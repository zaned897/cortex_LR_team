import unittest
import sys
sys.path.append('/home/app/NPDB')
from NPDB.main_npdb_cover_beta import run
from test.utils_test import Utils
from test.custom_assertions import CustomAssertions

class MainNPDBCoverBetaTest(unittest.TestCase, Utils, CustomAssertions):

    def __init__(self, *args, **kwargs):
        super(MainNPDBCoverBetaTest, self).__init__(*args, **kwargs)
        self.expected_data_path = "/home/app/test/resources/expected_data/"
        self.schema_path = "/home/app/test/resources/schemas/data_schema.json"

    def test_run(self):
        with self.get_files() as pdf_files:
            for pdf_file in pdf_files:
                expected_data_path = self.replace_ext(pdf_file)
                with self.open_json(expected_data_path) as expected_data:
                    result_model = run(self.pdf_file_path(pdf_file))
                    with self.subTest(msg="Checking pdf file", pdf_file=pdf_file.name, result_model=result_model, expected_data=expected_data):
                        self.assert_schema(result_model, pdf_file)
                        self.assert_fields(result_model, expected_data)
                        self.assertTrue(True)