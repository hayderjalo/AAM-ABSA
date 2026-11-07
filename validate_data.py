"""Validate user-supplied split CSVs without exporting their contents."""
import argparse
import csv
import json
from pathlib import Path


def validate(data_dir, image_dir):
    data_dir, image_dir = Path(data_dir), Path(image_dir).resolve()
    seen = set()
    counts = {}
    for split in ('train', 'validation', 'test'):
        path = data_dir / f'absa_{split}.csv'
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'sample_id', 'text', 'image_filename', 'aspect_labels_json'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{split}: required columns are {sorted(required)}')
            count = 0
            for row in reader:
                count += 1
                sid = row['sample_id'].strip()
                if not sid or sid in seen:
                    raise ValueError(f'{split}, row {count}: missing or repeated sample_id')
                seen.add(sid)
                if not row['text'].strip():
                    raise ValueError(f'{split}, row {count}: empty text')
                image = (image_dir / row['image_filename']).resolve()
                if not image.is_relative_to(image_dir) or not image.is_file():
                    raise ValueError(f'{split}, row {count}: invalid image path')
                labels = json.loads(row['aspect_labels_json'])
                if isinstance(labels, dict):
                    labels = labels.get('aspects', labels.get('labels'))
                if not isinstance(labels, list) or not labels:
                    raise ValueError(f'{split}, row {count}: expected nonempty label list')
                for label in labels:
                    if not isinstance(label, dict) or not label.get('aspect', label.get('category')):
                        raise ValueError(f'{split}, row {count}: missing aspect')
                    if label.get('sentiment', label.get('polarity')) not in ('negative', 'neutral', 'positive'):
                        raise ValueError(f'{split}, row {count}: invalid sentiment')
            if not count:
                raise ValueError(f'{split}: empty split')
            counts[split] = count
    print('Input structure validated:', counts)
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--image-dir', required=True)
    args = parser.parse_args()
    validate(args.data_dir, args.image_dir)
