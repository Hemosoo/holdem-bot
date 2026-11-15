"""Local table dashboard. python dashboard.py [--video PATH | --live]."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import sys
import time
import cv2
import numpy as np
from dashboard_logic import advice
from table_reader import TableReader
from table_locator import TableLocator


def tracking_summary(snapshot):
    if not snapshot:
        return 'Waiting for a completed reading.'
    fields=snapshot['fields']
    cards=sum(r.get('value') is not None for k,r in fields.items() if k.startswith('board_') or '.card_' in k)
    stacks=sum(r.get('value') is not None for k,r in fields.items() if k.endswith('.stack'))
    if stacks==0 and fields.get('pot',{}).get('value') is None:
        return 'Table not recognized in this frame. Check that the preview shows the game, then verify the table crop and layout.'
    if cards==0:
        return f'{stacks} stacks read; no cards identified. They may be hidden, not dealt, obscured, or outside the configured card regions.'
    return f'{cards} cards identified; {stacks} stacks read. Unknown fields remain unknown.'


class Dashboard:
    def __init__(self,source=None,live=False,layout='video',viewport=None,monitor=1):
        self.source=source or Path(__file__).parent/'fixtures/video-40.png'
        self.live=live
        self.monitor=monitor
        self.auto_viewport=live and viewport is None
        self.locator=TableLocator()
        self.stage="Waiting for capture"; self.started_at=None
        self.frame_size=None; self.preview_version=0; self.last_frame=None
        self.reader=TableReader(layout=layout,viewport=viewport)
        self.current_source=None; self.snapshot=None; self.preview=b''; self.busy=False; self.error=None
        self.lock=threading.Lock(); self.captured_at=0; self.source_seconds=None
        self.context={'hero':5,'seats':list(range(8)),'turn':'auto','street':'auto','last_to_act':'unknown','can_raise':False}

    def capture(self,seconds=0,uploaded=None,reuse=False,browser_share=False):
        if not isinstance(seconds,(int,float)) or not np.isfinite(seconds) or seconds<0: raise ValueError('Invalid video time.')
        with self.lock:
            if self.busy: return False
            if browser_share and self.current_source!='Browser screen share':
                self.reader.viewport=None
                self.auto_viewport=False
                self.reader.history.clear()
            self.busy=True; self.error=None; self.context.pop('confirmed_at',None)
            self.started_at=time.time(); self.stage='Capturing image'
            # Keep the last frame visible while the next capture is being obtained.
        def work():
            try:
                source_seconds=self.source_seconds if reuse else None
                if reuse:
                    frame=self.last_frame.copy() if self.last_frame is not None else None
                elif uploaded is not None:
                    frame=cv2.imdecode(np.frombuffer(uploaded,np.uint8),cv2.IMREAD_COLOR)
                elif self.live:
                    import mss
                    if sys.platform=='darwin':
                        try:
                            import Quartz
                        except ImportError:
                            Quartz=None
                        check=getattr(Quartz,'CGPreflightScreenCaptureAccess',None)
                        if check and not check():
                            raise RuntimeError('Screen Recording permission is required. In System Settings > Privacy & Security > Screen Recording (or Screen & System Audio Recording), allow the app running Python: Terminal, iTerm or VS Code. Quit and reopen that app, then restart this dashboard.')
                    with mss.MSS() as sct:
                        if not 1<=self.monitor<len(sct.monitors):
                            raise ValueError(f'Monitor {self.monitor} is unavailable. Available monitors: 1–{len(sct.monitors)-1}.')
                        frame=cv2.cvtColor(np.asarray(sct.grab(sct.monitors[self.monitor])),cv2.COLOR_BGRA2BGR)
                elif str(self.source).lower().endswith(('.mp4','.mov','.avi','.mkv')):
                    cap=cv2.VideoCapture(str(self.source))
                    try:
                        cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
                        ok,frame=cap.read()
                        if not ok: raise ValueError('No video frame at this time.')
                        source_seconds=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    finally: cap.release()
                else: frame=cv2.imread(str(self.source))
                if frame is None: raise ValueError('Cannot decode image.')
                if frame.shape[0]*frame.shape[1]>24_000_000: raise ValueError('Image is too large.')
                captured=self.captured_at if reuse else time.time()
                ok,jpeg=cv2.imencode('.jpg',cv2.resize(frame,(900,round(frame.shape[0]*900/frame.shape[1]))))
                if not ok: raise ValueError('Cannot encode preview.')
                with self.lock:
                    self.preview=jpeg.tobytes(); self.captured_at=captured
                    self.preview_version+=1; self.frame_size=[frame.shape[1],frame.shape[0]]
                    self.last_frame=frame; self.stage='Reading cards and text'
                    self.source_seconds=source_seconds
                    if not reuse: self.current_source=('Browser screen share' if browser_share else 'Uploaded image') if uploaded is not None else None
                if self.auto_viewport:
                    with self.lock: self.stage='Locating game window'
                    viewport=self.locator.locate(frame)
                    with self.lock: self.reader.viewport=viewport
                    if viewport is None:
                        raise RuntimeError('Could not locate the known game layout on this screen. Pause Auto update and drag around the entire game client in the preview, or use Share game window.')
                with self.lock: self.stage='Reading cards and text'
                snapshot=self.reader.read(frame)
                with self.lock:
                    self.snapshot=snapshot; self.stage='Reading complete'
            except Exception as exc:
                message=str(exc)
                if 'tesseract is not installed' in message.lower():
                    message+=' On macOS, run brew install tesseract, then restart the dashboard.'
                with self.lock: self.error=message; self.stage='Capture or recognition failed'
                print(f'Dashboard error: {message}',flush=True)
            finally:
                with self.lock: self.busy=False
        threading.Thread(target=work,daemon=True).start()
        return True

    def state(self):
        with self.lock:
            age=time.time()-self.captured_at if self.captured_at else None
            recommendation=advice(self.snapshot,self.context,stale=age is not None and age>30)
            if self.busy or self.error:
                recommendation.update(action=None,reason='Reading a new frame…' if self.busy else 'Capture failed. Retry before using a suggestion.')
            return {'snapshot':self.snapshot,'context':self.context.copy(),'advice':recommendation,
                    'busy':self.busy,'error':self.error,'age':age,
                    'tracking':tracking_summary(self.snapshot),'stage':self.stage,'elapsed':round(time.time()-self.started_at,1) if self.busy and self.started_at else None,
                    'preview_ready':bool(self.preview),'preview_version':self.preview_version,
                    'auto_viewport':self.auto_viewport,'frame_size':self.frame_size,'viewport':self.reader.viewport,'monitor':self.monitor,'source_seconds':self.source_seconds,
                    'source':self.current_source or ('Live capture' if self.live else str(self.source.name)), 'live':self.live}


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,data,status=200,kind='application/json'):
            body=json.dumps(data).encode() if kind=='application/json' else data
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=self.path.split('?')[0]
            if path=='/': self.send((Path(__file__).parent/'dashboard.html').read_bytes(),kind='text/html; charset=utf-8')
            elif path=='/api/state': self.send(app.state())
            elif path=='/api/preview':
                with app.lock: data=app.preview
                self.send(data,kind='image/jpeg') if data else self.send({'error':'No preview'},404)
            else: self.send({'error':'Not found'},404)
        def do_POST(self):
            # Prevent another website from issuing browser requests to this local service.
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
                return self.send({'error':'Origin rejected'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16*1024*1024: raise ValueError('Request must be between 1 byte and 16 MiB.')
                body=self.rfile.read(size)
                if self.path=='/api/image': return self.send({'started':app.capture(uploaded=body,browser_share=self.headers.get('X-Capture-Source')=='browser')},202)
                data=json.loads(body)
                if not isinstance(data,dict): raise ValueError('Expected a JSON object.')
                if self.path=='/api/capture': return self.send({'started':app.capture(data.get('seconds',0))},202)
                if self.path=='/api/viewport':
                    viewport=data.get('viewport')
                    with app.lock:
                        if app.busy: return self.send({'error':'Wait for the current reading to finish before changing the crop.'},409)
                        if app.last_frame is None: raise ValueError('Capture an image before selecting a crop.')
                        if viewport is not None:
                            if not isinstance(viewport,list) or len(viewport)!=4 or any(type(v) is not int for v in viewport): raise ValueError('Invalid crop rectangle.')
                            x,y,w,h=viewport
                            if min(x,y)<0 or min(w,h)<=0 or x+w>app.frame_size[0] or y+h>app.frame_size[1]: raise ValueError('Crop is outside the captured image.')
                        app.reader.viewport=viewport
                        app.auto_viewport=viewport is None and app.live and app.current_source!='Browser screen share'
                        app.reader.history.clear()
                    return self.send({'started':app.capture(reuse=True)},202)
                if self.path=='/api/context':
                    with app.lock:
                        if app.busy: return self.send({'error':'Wait for the current reading.'},409)
                        allowed={'hero','seats','dealer','turn','street','call','raises','aggressor','last_to_act','can_raise'}
                        updated={k:v for k,v in data.items() if k in allowed}
                        merged=app.context|updated
                        if data.get('confirm') and app.snapshot: merged['confirmed_at']=app.snapshot['timestamp']
                        else: merged.pop('confirmed_at',None)
                        advice(app.snapshot,merged)
                        app.context=merged
                    return self.send({'saved':True})
                self.send({'error':'Not found'},404)
            except (ValueError,TypeError,KeyError,IndexError) as exc:
                self.send({'error':str(exc)},400)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group();source.add_argument('--video',type=Path);source.add_argument('--image',type=Path);source.add_argument('--live',action='store_true')
    p.add_argument('--layout',choices=['video','compact','expanded','auto'],default='video')
    p.add_argument('--viewport',nargs=4,type=int)
    p.add_argument('--monitor',type=int,default=1,help='Live display number (default: 1)')
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    app=Dashboard(args.video or args.image,args.live,args.layout,args.viewport,args.monitor)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(app))
    app.capture()
    print(f'Dashboard listening on port {args.port}. Press Ctrl+C to stop.',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__': main()
