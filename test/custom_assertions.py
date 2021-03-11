import jsonschema

FIELDS = [
  "license",
  "npi",
  "dcn",
  "entity_name",
  "practitioner_name",
  "action_initial",
  "action_basis",
  "event_outcome",
  "process_date",
  "payment_date",
  "event_day",
  "amount_pract",
  "payment_total_amount"
]

class CustomAssertions(object):
    """docstring for CustomAssertions"""
    def __init__(self, arg):
        super(CustomAssertions, self).__init__()
    
    def assert_schema(self, result, pdf_file):
        with self.open_json(self.schema_path) as schema:
            try:
                print("Valid schema for this file: %s ..." % pdf_file.name)
                return self.assertTrue(True)
            except jsonschema.exceptions.ValidationError as validation_error:
                return self.assertTrue(False)

    def assert_fields(self, results, expected_data):
        print("Validating fields...")
        for result, expected in zip(results, expected_data):
          for field in FIELDS:
              try:
                  self.assertTrue(result[field] == expected[field])
              except KeyError as e:
                  self.assertTrue(False)
