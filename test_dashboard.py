import copy
import threading
import time
import types
from unittest.mock import patch, MagicMock
import numpy as np
from dashboard import Dashboard
import unittest
from dashboard_logic import positions, advice


def snapshot(cards=('As','Ah'),board=(),dealer=0):
    fields={'dealer_seat':{'value':dealer},'seat_5.stack':{'value':100},'pot':{'value':5},'seat_5.turn_ring':{'value':True}}
    fields.update({f'seat_5.card_{i}':{'value':c} for i,c in enumerate(cards)})
    fields.update({f'board_{i}':{'value':board[i] if i<len(board) else None} for i in range(5)})
    return {'timestamp':123,'fields':fields,'table':{'street':{0:None,3:'flop',4:'turn',5:'river'}[len(board)]}}


def context(**kwargs):
    return dict(hero=5,seats=list(range(8)),turn='auto',street='preflop',call=0,raises=0,last_to_act='no',can_raise=True,confirmed_at=123,**kwargs)


class DashboardTests(unittest.TestCase):
    def test_positions_rotate_skip_empty_keep_folded(self):
        self.assertEqual(positions(list(range(8)),7)[5],'HJ')
        self.assertEqual(positions([0,2,4,5,6,7],7),{7:'BTN',0:'SB',2:'BB',4:'UTG',5:'HJ',6:'CO'})
        self.assertEqual(positions([2,5],5),{5:'BTN / SB',2:'BB'})
        self.assertEqual(positions(list(range(8)),0)[7],'CO')
        self.assertEqual(positions([1,2],0),{})

    def test_aces_open_and_free_check_no_illegal_fold(self):
        self.assertEqual(advice(snapshot(),context())['action'],'Bet')
        self.assertEqual(advice(snapshot(('7s','2h')),context())['action'],'Check')

    def test_turn_stale_and_confirmation_gates(self):
        s=snapshot();c=context()
        self.assertIsNone(advice(s,c,stale=True)['action'])
        c['confirmed_at']=122;self.assertIsNone(advice(s,c)['action'])
        c['confirmed_at']=123;s['fields']['seat_5.turn_ring']['value']=None
        self.assertIsNone(advice(s,c)['action'])

    def test_missing_amount_board_duplicate(self):
        s=snapshot();c=context();c['call']=None
        self.assertIsNone(advice(s,c)['action'])
        c=context();s=snapshot(('As','As'));self.assertIsNone(advice(s,c)['action'])
        s=snapshot(board=('2s','3d','4c'));self.assertIsNone(advice(s,context())['action'])

    def test_postflop_policy_with_confirmed_call(self):
        s=snapshot(('Jd','5d'),('Js','3c','2h'));c=context();c.update(street='auto',call=10.5)
        self.assertEqual(advice(s,c)['action'],'Fold')
        c['can_raise']=False;c['call']=0
        self.assertEqual(advice(s,c)['action'],'Check')

    def test_invalid_context_rejected_before_gates(self):
        for change in ({'call':float('nan')},{'hero':9},{'raises':-1},{'seats':[True]},{'turn':'bad'}):
            c=context();c.update(change)
            with self.assertRaises(ValueError): advice(None,c)


class CaptureTests(unittest.TestCase):
    def wait_done(self,app):
        deadline=time.monotonic()+3
        while app.busy and time.monotonic()<deadline: time.sleep(.01)
        self.assertFalse(app.busy)

    def test_preview_published_before_ocr_and_modern_capture_api(self):
        started=threading.Event();release=threading.Event()
        def read(frame):
            started.set();release.wait(3);return snapshot()
        with patch('dashboard.TableReader') as reader, patch('mss.MSS') as capture:
            reader.return_value.read.side_effect=read
            sct=capture.return_value.__enter__.return_value
            sct.monitors=[{}, {'left':0,'top':0,'width':30,'height':20}]
            sct.grab.return_value=np.zeros((20,30,4),np.uint8)
            app=Dashboard(live=True);app.capture()
            try:
                self.assertTrue(started.wait(2))
                state=app.state()
                self.assertTrue(state['preview_ready'])
                self.assertTrue(state['busy'])
                self.assertIsNone(state['snapshot'])
                self.assertEqual(state['frame_size'],[30,20])
                self.assertIsNone(state['advice']['action'])
                capture.assert_called_once()
            finally: release.set();self.wait_done(app)
            self.assertIsNone(app.error)

    def test_macos_permission_failure_is_actionable(self):
        quartz=types.SimpleNamespace(CGPreflightScreenCaptureAccess=lambda:False)
        with patch('dashboard.TableReader'), patch('dashboard.sys.platform','darwin'), patch.dict('sys.modules',{'Quartz':quartz}):
            app=Dashboard(live=True);app.capture();self.wait_done(app)
            self.assertIn('Screen Recording',app.error)
            self.assertIn('Quit and reopen',app.error)
            self.assertFalse(app.state()['preview_ready'])

    def test_ocr_error_keeps_preview_and_reports_failure(self):
        with patch('dashboard.TableReader') as reader:
            reader.return_value.read.side_effect=RuntimeError('Tesseract is not installed or not in PATH')
            app=Dashboard();app.capture();self.wait_done(app)
            state=app.state()
            self.assertTrue(state['preview_ready'])
            self.assertIn('brew install tesseract',state['error'])
            self.assertIsNone(state['advice']['action'])


if __name__=='__main__': unittest.main()
