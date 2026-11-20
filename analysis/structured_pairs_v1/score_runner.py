"""Load the missing inference symbol without changing the frozen holdout source."""
import holdout
from gridpath.planning.verification import Verifier

holdout.Verifier = Verifier
result = holdout.score()
print("challenge", result["challenge_sha256"], "G", result["generated_sha256"])
for name, model in result["models"].items():
    for panel in ("controlled", "natural_G"):
        cases = model[panel]["cases"]
        print(name, panel,
              sum(case["prediction"]["text"] == case["row"]["target_text"]
                  for case in cases), "/", len(cases),
              "F1", model[panel]["metrics"]["macro_f1"],
              "V_failures", model[panel]["metrics"]["interface_failure_count"])
