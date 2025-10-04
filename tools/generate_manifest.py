import csv, json, os, glob, hashlib

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:12]

repo = {"csv_headers":{}, "files":[]}

for p in ["app/shots.csv", "data/videos.csv"]:
    if os.path.exists(p):
        with open(p, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, [])
        repo["csv_headers"][p] = headers

for pattern in ["app/**/*.py", "src/**/*.py", "data/homography/*.json", "app/reports/**/*.*"]:
    for path in glob.glob(pattern, recursive=True):
        if os.path.isfile(path):
            repo["files"].append({"path": path, "sha": sha(path)})

print(json.dumps(repo, indent=2))
