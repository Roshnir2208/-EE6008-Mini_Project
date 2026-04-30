from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

# CHANGE THIS to the extracted Roboflow Pascal VOC dataset folder
SOURCE_ROOT = Path(r"C:\Users\sosih\OneDrive\Desktop\Self_Improvement\MEng\Deep_Learning_At_The_Edge\Final_work\-EE6008-Mini_Project\Dataset\Garbage Classifier.v27i.voc")

# Output folder
DEST_ROOT = Path(r"C:\Users\sosih\OneDrive\Desktop\Self_Improvement\MEng\Deep_Learning_At_The_Edge\Final_work\-EE6008-Mini_Project\Dataset\Modified_Dataset")

# merge confusing classes
MERGE_LABELS = {
    "Garbage": "Trash",
    "garbage": "Trash",
    "trash": "Trash",
    "Cardboard": "Paper",   # Trying to merge cardboard and paper
    "cardboard": "Paper",
}

# Keep only these final classes
KEEP_CLASSES = {"Glass", "Metal", "Paper", "Plastic", "Trash"}

SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

def normalise_label(label: str) -> str:
    label = label.strip()
    return MERGE_LABELS.get(label, label)

def parse_labels(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name:
            labels.append(normalise_label(name))

    return labels

def copy_pair(img_path: Path, xml_path: Path, split: str, final_label: str):
    out_dir = DEST_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rename to include label for easier checking
    safe_stem = f"{final_label}_{img_path.stem}"
    out_img = out_dir / f"{safe_stem}{img_path.suffix.lower()}"
    out_xml = out_dir / f"{safe_stem}.xml"

    shutil.copy2(img_path, out_img)
    shutil.copy2(xml_path, out_xml)

    # Update XML filename and label after merging
    tree = ET.parse(out_xml)
    root = tree.getroot()

    filename_node = root.find("filename")
    if filename_node is not None:
        filename_node.text = out_img.name

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is not None:
            name_node.text = final_label

    tree.write(out_xml)

def main():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "kept": 0,
        "skipped_multi_label": 0,
        "skipped_no_xml": 0,
        "skipped_class": 0,
    }

    for split in SPLITS:
        split_dir = SOURCE_ROOT / split
        if not split_dir.exists():
            print(f"Skipping missing split: {split_dir}")
            continue

        for img_path in split_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            xml_path = img_path.with_suffix(".xml")

            if not xml_path.exists():
                summary["skipped_no_xml"] += 1
                continue

            labels = parse_labels(xml_path)
            unique_labels = sorted(set(labels))

            # Keep only images with exactly one unique label
            if len(unique_labels) != 1:
                summary["skipped_multi_label"] += 1
                continue

            final_label = unique_labels[0]

            if final_label not in KEEP_CLASSES:
                summary["skipped_class"] += 1
                continue

            copy_pair(img_path, xml_path, split, final_label)
            summary["kept"] += 1

    print("\nDone.")
    print(f"Output: {DEST_ROOT}")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
