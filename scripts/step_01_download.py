import os, json
from roboflow import Roboflow

ROBOFLOW_API_KEY = "mFcoakFBfID8vndXE5gS"
WORKSPACE        = "student-utr07"
PROJECT          = "garbage-classifier-oehkt"
VERSION          = 27
RAW_DIR          = "raw_dataset"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("voc", location=RAW_DIR)

import glob, xml.etree.ElementTree as ET

classes = set()
for xml_path in glob.glob(os.path.join(RAW_DIR, "**", "*.xml"), recursive=True):
    tree = ET.parse(xml_path)
    for obj in tree.findall("object"):
        classes.add(obj.find("name").text.strip().lower())

CLASS_NAMES = sorted(classes)
label_map = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
label_map[0] = "background"

with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print(f"Found {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
print("label_map.json written.")

# Checkpoint: must be exactly 8 entries
assert len(label_map) == 8, f"Expected 8 label_map entries, got {len(label_map)}"
print(f"\nCHECKPOINT PASS: label_map.json has exactly {len(label_map)} entries")

for split in ["train", "valid", "test"]:
    xmls = glob.glob(os.path.join(RAW_DIR, split, "Annotations", "*.xml"))
    counts = {}
    for xp in xmls:
        for obj in ET.parse(xp).findall("object"):
            n = obj.find("name").text.strip().lower()
            counts[n] = counts.get(n, 0) + 1
    print(f"\n{split} ({len(xmls)} images):")
    for cls in CLASS_NAMES:
        print(f"  {cls:12s}: {counts.get(cls, 0)} objects")
