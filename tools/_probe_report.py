import json

with open("/home/wf/radarNWPQML/runs/qlfn2_pilot/eval_test.json", encoding="utf-8") as handle:
    d = json.load(handle)

print("top keys:", list(d.keys()))
print("has thresholds:", "thresholds" in d)
if "thresholds" in d:
    print("threshold keys:", list(d["thresholds"].keys()))
