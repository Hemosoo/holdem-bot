"""Functional vision checks using generated fixtures; no desktop required."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from table_reader import TableReader, SuitMatcher, Reading, parse_amount, text_read, turn_ring, summarize, locate_board

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

class RecognitionTests(unittest.TestCase):
    def test_amounts(self):
        for raw, expected in [('POT: 6.50 BB',6.5),('450.50 BB',450.5),('450.50BB',450.5),('$1,234.50',1234.5),('1 2',None),('Waiting',None),('',None)]:
            self.assertEqual(parse_amount(raw),expected,raw)

    def test_suit_shapes(self):
        matcher=SuitMatcher()
        for suit,glyph in zip('cdhs','♣♦♥♠'):
            image=Image.new('RGB',(70,90),'white')
            ImageDraw.Draw(image).text((5,0),glyph,font=ImageFont.truetype(FONT,65),fill='red' if suit in 'dh' else 'black')
            result,confidence=matcher.read(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
            self.assertEqual(result,suit)
            self.assertGreater(confidence,.65)

    def test_ten_of_hearts(self):
        image=Image.new('RGB',(64,91),'white')
        draw=ImageDraw.Draw(image)
        draw.text((1,-3),'10',font=ImageFont.truetype(FONT,25),fill='red')
        draw.text((1,35),'♥',font=ImageFont.truetype(FONT,22),fill='red')
        result=TableReader().card(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR),(0,0,64,91))
        self.assertEqual(result.value,'Th',result)

    def test_real_ocr_amount(self):
        image=Image.new('RGB',(280,50),'#20252b')
        ImageDraw.Draw(image).text((5,2),'POT: 6.50 BB',font=ImageFont.truetype(FONT,24),fill='white')
        r=text_read(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR),(0,0,280,50),True)
        self.assertEqual(r.value,6.5,r.raw)

    def test_temporal_confirmation_and_removal(self):
        reader=TableReader(confirmations=2)
        frame=np.zeros((765,1024,3),np.uint8)
        with patch('table_reader.text_read',return_value=Reading(100,.9)), patch('table_reader.amount_read',side_effect=lambda *args:Reading(100,.9)):
            self.assertNotIn('pot',reader.read(frame)['confirmed'])
            self.assertEqual(reader.read(frame)['confirmed']['pot'],100)
        with patch('table_reader.text_read',return_value=Reading()), patch('table_reader.amount_read',side_effect=lambda *args:Reading()):
            self.assertNotIn('pot',reader.read(frame)['confirmed'])

    def test_empty_frame_does_not_invent_cards(self):
        reader=TableReader()
        with patch('table_reader.text_read',return_value=Reading()), patch('table_reader.amount_read',side_effect=lambda *args:Reading()):
            for h in (677,765,1530):
                result=reader.read(np.zeros((h,1024 if h<1000 else 2048,3),np.uint8))
                self.assertTrue(all(r['value'] is None for r in result['fields'].values()))

    def test_turn_ring_is_positive_evidence_only(self):
        image=np.zeros((100,100,3),np.uint8)
        self.assertIsNone(turn_ring(image,(50,50)).value)
        cv2.circle(image,(50,50),32,(255,140,45),4)
        self.assertIs(turn_ring(image,(50,50)).value,True)

    def test_status_and_partial_board(self):
        fields={'seat_4.status':Reading('SITTING OUT',.9),
                'seat_3.avatar_text':Reading('Waiting',.9),
                'board_0':Reading('Jc',.9),'board_2':Reading('Th',.9)}
        result=summarize(fields)
        self.assertEqual(result['players'][4]['status'],'sitting_out')
        self.assertEqual(result['players'][3]['status'],'waiting')
        self.assertIsNone(result['players'][0]['status'])
        self.assertIsNone(result['street'])
        fields['board_1']=Reading('9c',.9)
        self.assertEqual(summarize(fields)['street'],'flop')

    def test_all_duplicate_cards_rejected(self):
        reader=TableReader()
        with patch('table_reader.text_read',return_value=Reading()), patch.object(reader,'card',side_effect=lambda *args:Reading('Ah',.9)):
            result=reader.read(np.zeros((765,1024,3),np.uint8))
        self.assertTrue(all(result['fields'][f'board_{i}']['value'] is None for i in range(5)))
        self.assertTrue(all(p['cards']==[None,None] for p in result['table']['players']))

    def test_real_board_under_resizing_and_dimming(self):
        image=cv2.imread(str(Path(__file__).parent/'fixtures/ChatGPT Image Nov 11, 2026, 10_12_15 PM.png'))
        self.assertIsNotNone(image)
        reader=TableReader()
        for factor in (1.0,.7):
            for x,label in [(361,'Jc'),(426,'9c'),(491,'Th'),(556,'2c'),(621,'2h')]:
                crop=image[292:383,x:x+64]
                crop=cv2.resize(cv2.resize(crop,(48,68)),(64,91))
                crop=np.clip(crop.astype(float)*factor,0,255).astype('uint8')
                self.assertEqual(reader.card(crop,(0,0,64,91)).value,label,(factor,label))

    def test_original_calibration_frames(self):
        root=Path(__file__).parent
        examples=json.loads((root/'fixtures/table_examples.json').read_text())['examples']
        for example in examples:
            frame=cv2.imread(str(root/'fixtures'/example['image']))
            result=TableReader(layout=example['layout'],confirmations=1).read(frame)
            if example['layout']=='expanded':
                self.assertEqual(result['table']['players'][4]['status'],'sitting_out')
                for j in range(2):
                    field=result['fields'][f'seat_6.card_{j}']
                    self.assertIsNone(field['value'])
                    self.assertEqual(field['candidate'],'7s')
            else:
                self.assertEqual(result['table']['players'][0]['status'],'raise')
            for key,expected in example['expected'].items():
                actual=result['fields'][key]
                if example['layout']=='compact' and key=='seat_0.stack':
                    self.assertIsNone(actual['value'])
                    self.assertIn('Conflicting OCR',actual['raw'])
                else:
                    self.assertEqual(actual['value'],expected,(example['layout'],key,actual))

    def test_board_localization_spacing(self):
        for count in (3,4,5):
            image=np.zeros((684,1024,3),np.uint8)
            for i in range(count):
                cv2.rectangle(image,(330+i*63,255),(389+i*63,337),(245,245,245),-1)
            self.assertEqual(len(locate_board(image)),count)
        image=np.zeros((684,1024,3),np.uint8)
        self.assertEqual(locate_board(image),[])

    def test_video_flop_then_new_hand(self):
        root=Path(__file__).parent/'fixtures'
        reader=TableReader(layout='video')
        flop=reader.read(cv2.imread(str(root/'video-40.png')))
        self.assertEqual(flop['table']['board'],['Js','3c','2h',None,None])
        self.assertEqual(flop['table']['pot'],31)
        self.assertEqual(flop['table']['players'][5]['cards'],['Jd','5d'])
        self.assertEqual(flop['table']['players'][5]['stack'],194)
        self.assertEqual([p['stack'] for p in flop['table']['players']], [195.5,160.5,175.5,70,809.5,194,199,370.5])
        reset=reader.read(cv2.imread(str(root/'video-70.png')))
        self.assertEqual(reset['table']['board'],[None]*5)
        self.assertEqual(reset['table']['players'][5]['cards'],[None]*2)
        self.assertEqual(reset['table']['pot'],1.5)
        self.assertEqual(reset['table']['players'][5]['stack'],63.5)
        self.assertFalse(any(k.startswith('board_') for k in reset['confirmed']))

    def test_viewport_validation(self):
        reader=TableReader(viewport=(100,100,200,200))
        with self.assertRaises(ValueError):
            reader.read(np.zeros((100,100,3),np.uint8))

if __name__=='__main__': unittest.main()
