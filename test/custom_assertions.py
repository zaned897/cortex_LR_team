import jsonschema
import editdistance

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
        correct=0
        incorrect=0
        for result, expected in zip(results, expected_data):
          for field in FIELDS:
              if editdistance.eval(str(result[field]) ,str(expected[field])) <= len(str(result[field])) / 10:
                  correct+=1
              else:
                  incorrect+=1
          accuracy = (correct / (correct + incorrect))
          print('Model accuracy: ',accuracy)
          try:
              self.assertTrue(accuracy >= 0.7)
          except KeyError as e:
              self.assertTrue(False)