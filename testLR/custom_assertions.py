import jsonschema
import editdistance

FIELDS = [
    "claim",
    "status",
    "insured",
    "claimant",
    "as_of",
    "loss_date",
    "report_date",
    "closed_date",
    "exp_date",
    "eff_date",
    "inc_date",
    "status_date",
    "state",
    "total_paid",
    "exps_paid",
    "exps_reserve",
    "exps_incurred",
    "indemnity_paid",
    "indemnity_reserve",
    "indemnity_incurred",
    "total_incurred",
    "total_reserve",
    "incident_desc"

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
            try:
                if editdistance.eval(str(result[field]) ,str(expected[field])) <= len(str(result[field])) / 10:
                    correct+=1
                else:
                    incorrect+=1
            except:
                pass
          accuracy = (correct / (correct + incorrect))
          print('Model accuracy: ',accuracy)
          try:
              self.assertTrue(accuracy >= 0.7)
          except KeyError as e:
              self.assertTrue(False)
