import os, sys, collections, zipfile

B = "scratchpad/row55_l2i8s4e1c_bundle"
n = 0
tot = 0
per = collections.Counter()
for root, dirs, files in os.walk(B):
    for f in files:
        p = os.path.join(root, f)
        n += 1
        tot += os.path.getsize(p)
        rel = os.path.relpath(p, B).replace(os.sep, "/")
        per[rel.split("/")[0]] += 1
print("on disk: files", n, "bytes", tot)
for k in sorted(per):
    print("   ", k, per[k])

z = zipfile.ZipFile(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/row55_l2i8s4e1c_bundle.zip")
zn = [x for x in z.namelist() if not x.endswith("/")]
disk = set()
for root, dirs, files in os.walk(B):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), B).replace(os.sep, "/")
        disk.add("row55_l2i8s4e1c_bundle/" + rel)
zs = set(zn)
print("zip: files", len(zn))
print("in zip not on disk:", sorted(zs - disk)[:10])
print("on disk not in zip:", sorted(disk - zs)[:10])
