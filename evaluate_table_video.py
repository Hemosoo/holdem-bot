"""Sample and evaluate the user-supplied 71-second test clip against manual labels.

This evaluator labels nine instants, not all frames or events in the recording.
40s and 50s supplied additional glyph calibration; others are regression samples.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
from table_reader import TableReader, annotate

SAMPLES=[
    (0,3.5,199,None,None), (5,3.5,199,None,['Jd','5d']),
    (10,9.5,199,None,['Jd','5d']), (20,20.5,194,None,['Jd','5d']),
    (30,20.5,194,None,['Jd','5d']), (40,31,194,['Js','3c','2h'],['Jd','5d']),
    (50,83.5,162.5,['Js','3c','2h','Ad','Qd'],['Jd','5d']),
    (60,83.5,162.5,['Js','3c','2h','Ad','Qd'],['Jd','5d']),
    (70,1.5,63.5,None,None),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--output',type=Path,default=Path('/tmp/table-video-evaluation'))
    args=parser.parse_args()
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened(): parser.error('Cannot open video')
    args.output.mkdir(parents=True,exist_ok=True)
    reader=TableReader(layout='video',confirmations=2)
    report={'coverage':'Nine labeled sample frames only; not exhaustive video accuracy',
            'fps':cap.get(cv2.CAP_PROP_FPS),'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'samples':[],'checks':0,'mismatches':0}
    try:
        for sec,pot,stack,board,hero in SAMPLES:
            cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Cannot decode frame at {sec}s')
            decoded_seconds=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            start=time.monotonic();result=reader.read(frame)
            elapsed=time.monotonic()-start
            result.update(source_seconds=decoded_seconds,requested_seconds=sec)
            expected={'pot':pot,'seat_5.stack':stack}
            expected.update({f'board_{i}':(board or [])[i] if i<len(board or []) else None for i in range(5)})
            expected.update({f'seat_5.card_{i}':(hero or [None,None])[i] for i in range(2)})
            mismatches=[]
            for key,value in expected.items():
                actual=result['fields'][key]
                if actual['value']!=value:
                    mismatches.append({'field':key,'expected':value,'actual':actual})
            report['checks']+=len(expected);report['mismatches']+=len(mismatches)
            sample={'seconds':sec,'decoded_seconds':decoded_seconds,'checks':len(expected),
                    'processing_seconds':round(elapsed,3),'mismatches':mismatches,
                    'board':result['table']['board'],'hero':result['table']['players'][5]['cards'],
                    'pot':result['table']['pot'],'hero_stack':result['table']['players'][5]['stack']}
            report['samples'].append(sample)
            (args.output/f'{sec:02}-observations.json').write_text(json.dumps(result,indent=2))
            cv2.imwrite(str(args.output/f'{sec:02}-overlay.png'),annotate(frame,result))
            print(json.dumps(sample),flush=True)
    finally:
        cap.release()
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'checks':report['checks'],'mismatches':report['mismatches']}))
    return int(report['mismatches']>0)


if __name__=='__main__':
    raise SystemExit(main())
