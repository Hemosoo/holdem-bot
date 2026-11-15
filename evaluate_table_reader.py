"""Evaluate original screenshots against human labels; missing files fail explicitly."""
import argparse
import json
from pathlib import Path
import cv2
from table_reader import TableReader, annotate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_directory',type=Path)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).parent/'fixtures/table_examples.json')
    parser.add_argument('--output',type=Path,default=Path('/tmp/table-evaluation'))
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    failures=0
    for index,example in enumerate(manifest['examples']):
        path=args.image_directory/example['image']
        if not path.is_file():
            report.append({'image':str(path),'error':'Original image file is not accessible; not evaluated'})
            failures+=1
            continue
        image=cv2.imread(str(path))
        if image is None:
            report.append({'image':str(path),'error':'Cannot decode image'})
            failures+=1
            continue
        result=TableReader(layout=example['layout'],confirmations=1).read(image)
        mismatches=[]
        for field,expected in example['expected'].items():
            actual=result['fields'][field]['value']
            if actual!=expected:
                mismatches.append({'field':field,'expected':expected,'actual':actual,'raw':result['fields'][field]['raw']})
        failures+=len(mismatches)
        cv2.imwrite(str(args.output/f'{index}-overlay.png'),annotate(image,result))
        (args.output/f'{index}-observations.json').write_text(json.dumps(result,indent=2))
        report.append({'image':str(path),'checks':len(example['expected']),'mismatches':mismatches})
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
