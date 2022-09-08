import gzip
import json


def load_json_gz(filename):
    with gzip.open(filename, "rt") as f:
        return json.loads(f.read())


def dump_json_gz(filename, obj):
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)
