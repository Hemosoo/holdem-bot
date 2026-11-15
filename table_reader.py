"""Eight-seat table recognition. Output observations, never betting instructions.

Coordinates describe the supplied table skin, not arbitrary poker clients.
Unknown/obscured fields stay null. Confidence is a heuristic, not a probability.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Reading:
    value: object = None
    confidence: float = 0.0
    raw: str = ''
    bbox: tuple = ()
    candidate: object = None  # retained when consistency validation rejects a reading


def parse_amount(text):
    """Require one amount; never concatenate unrelated OCR tokens."""
    text = re.sub(r'(?i)\bpot\b|bb\b|[$£€:]', ' ', text)
    tokens = re.findall(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])', text)
    if len(tokens) != 1:
        return None
    try:
        return float(tokens[0].replace(',', ''))
    except ValueError:
        return None


def text_read(image, bbox, numeric=False):
    x, y, w, h = bbox
    crop = image[y:y+h, x:x+w]
    if not crop.size:
        return Reading(bbox=bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Text on this skin is light on a dark panel. Preserve punctuation.
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    data = pytesseract.image_to_data(gray, config='--psm 7', output_type=pytesseract.Output.DICT)
    words = [(t.strip(), float(c)) for t, c in zip(data['text'], data['conf']) if t.strip() and float(c) >= 0]
    raw = ' '.join(t for t, _ in words)
    conf = min((c for _, c in words), default=0) / 100
    value = parse_amount(raw) if numeric else (raw or None)
    if conf < .45:
        value = None
    return Reading(value, conf if value is not None else 0, raw, bbox)


def amount_read(image, bbox):
    """Separate large amount glyphs from the smaller BB suffix and avatar art."""
    x,y,w,h=bbox
    crop=image[y:y+h,x:x+w]
    if not crop.size: return Reading(bbox=bbox)
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask[hsv[:,:,1]>60]=0
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    candidates=[tuple(r) for r in stats[1:] if r[0]<55 and 8<=r[3]<=16 and r[4]>=8]
    if not candidates: return Reading(bbox=bbox)
    candidates.sort(key=lambda r:r[0])
    # The amount is the first text run; icons below it are excluded by baseline.
    first=candidates[0]
    line=[r for r in candidates if abs((r[1]+r[3])-(first[1]+first[3]))<=2]
    tallest=max(r[3] for r in line)
    digits=[r for r in line if r[3]>=tallest*.84]
    right=max(r[0]+r[2] for r in digits)
    top=min(r[1] for r in digits);bottom=max(r[1]+r[3] for r in digits)
    cut=gray[max(0,top-2):min(h,bottom+2),:min(w,right+2)]
    cut=cv2.resize(cut,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
    _,cut=cv2.threshold(cut,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cut=cv2.copyMakeBorder(cut,15,15,15,15,cv2.BORDER_CONSTANT,value=255)
    # Retry a second segmentation path and unrestricted OCR. The BB suffix has
    # already been removed, so it cannot be hallucinated as extra digits.
    binary=255-mask[max(0,top-1):min(h,bottom+1),:min(w,right+1)]
    binary=cv2.resize(binary,None,fx=5,fy=5)
    binary=cv2.copyMakeBorder(binary,15,15,15,15,cv2.BORDER_CONSTANT,value=255)
    observations=[]
    for prepared,config in [(cut,'--psm 7'),(binary,'--psm 7 -c tessedit_char_whitelist=0123456789.')]:
        data=pytesseract.image_to_data(prepared,config=config,output_type=pytesseract.Output.DICT)
        words=[(t,float(c)) for t,c in zip(data['text'],data['conf']) if t.strip() and float(c)>=0]
        raw=' '.join(t for t,_ in words)
        value=parse_amount(raw)
        conf=min((c for _,c in words),default=0)/100
        observations.append((conf,value,raw))
    reliable=[r for r in observations if r[0]>=.45 and r[1] is not None]
    if not reliable and observations[0][1] is not None and observations[0][1]==observations[1][1]:
        return Reading(observations[0][1],.45,'Agreement: '+observations[0][2],bbox)
    if not reliable:
        return Reading(raw=' | '.join(r[2] for r in observations),bbox=bbox)
    if len({r[1] for r in reliable})>1:
        ranked=sorted(reliable,key=lambda r:r[0],reverse=True)
        if ranked[0][0]-ranked[1][0]<.15:
            return Reading(raw='Conflicting OCR: '+' | '.join(r[2] for r in reliable),bbox=bbox)
        reliable=ranked[:1]
    conf,value,raw=max(reliable,key=lambda r:r[0])
    return Reading(value,conf,raw,bbox)


def chip_amount(image, search_box):
    """Use the colored chip icon to find the amount's actual starting column."""
    x,y,w,h=search_box
    crop=image[y:y+h,x:x+w]
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([65,60,50]),np.array([100,255,255]))
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    coins=[r for r in stats[1:] if 6<=r[2]<=19 and 6<=r[3]<=19 and r[4]>=12 and .6<r[2]/r[3]<1.6]
    if not coins: return Reading(bbox=search_box)
    # Filled centers and partial icons can be separate components of one ring.
    cx,cy,cw,ch,_=max(coins,key=lambda r:r[4])
    left=x+int(cx+cw)+3
    return amount_read(image,(left,y,max(1,x+w-left),h))


def turn_ring(image, center):
    """Only positive evidence: a blue ring, never an inferred turn from a timer."""
    x,y=center
    crop=image[max(0,y-39):y+40,max(0,x-39):x+40]
    if crop.shape[:2] != (79,79):
        return Reading()
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    yy,xx=np.ogrid[:79,:79]
    radius=np.sqrt((xx-39)**2+(yy-39)**2)
    annulus=(radius>=28)&(radius<=38)
    blue=(hsv[:,:,0]>=95)&(hsv[:,:,0]<=125)&(hsv[:,:,1]>90)&(hsv[:,:,2]>130)&annulus
    fraction=np.count_nonzero(blue)/np.count_nonzero(annulus)
    return Reading(True,min(.95,fraction),bbox=(x-39,y-39,79,79)) if fraction>.22 else Reading(bbox=(x-39,y-39,79,79))


def summarize(fields):
    """Expose structured players without interpreting unknown as absent or folded."""
    def value(key):
        return fields[key].value if key in fields else None
    players=[]
    for i in range(8):
        prefix=f'seat_{i}.'
        texts=[str(value(prefix+k) or '').lower() for k in ('status','avatar_text')]
        status=None
        for needle,label in [('sitting out','sitting_out'),('waiting','waiting'),('all in','all_in'),('all-in','all_in'),('raise','raise'),('fold','fold'),('check','check'),('call','call')]:
            if any(needle in text for text in texts):
                status=label
                break
        players.append({'seat':i,'name':value(prefix+'name'),'stack':value(prefix+'stack'),
                        'bet':value(prefix+'bet'),'status':status,
                        'turn_ring':value(prefix+'turn_ring'),
                        'cards':[value(prefix+f'card_{j}') for j in range(2)]})
    board=[value(f'board_{i}') for i in range(5)]
    # Partial/occluded cards cannot reliably establish the street.
    present=[v is not None for v in board]
    count=sum(present)
    street={3:'flop',4:'turn',5:'river'}.get(count) if present==([True]*count+[False]*(5-count)) else None
    return {'players':players,'board':board,'street':street,'pot':value('pot'),
            'display_units':'BB','dealer_seat':value('dealer_seat')}


def normalized_mask(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return np.zeros((48, 40), np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    roi = mask[y:y+h, x:x+w]
    scale = min(36 / w, 44 / h)
    roi = cv2.resize(roi, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_NEAREST)
    result = np.zeros((48, 40), np.uint8)
    yy, xx = (48-roi.shape[0])//2, (40-roi.shape[1])//2
    result[yy:yy+roi.shape[0], xx:xx+roi.shape[1]] = roi
    return result


class SuitMatcher:
    def __init__(self):
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        if not Path(font_path).exists():
            raise RuntimeError('Suit templates require DejaVuSans.ttf (fonts-dejavu-core on Linux).')
        self.templates = {}
        for suit, glyph in zip('cdhs', '♣♦♥♠'):
            im = Image.new('L', (100, 100))
            ImageDraw.Draw(im).text((5, 0), glyph, font=ImageFont.truetype(font_path, 72), fill=255)
            self.templates[suit] = normalized_mask((np.array(im) > 100).astype('uint8')*255)

    def read(self, image):
        if image.size == 0:
            return None, 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = ((hsv[:,:,0] < 15) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] > 65)
        mask = ((red | (gray < 115)).astype('uint8') * 255)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            return None, 0
        component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[component, cv2.CC_STAT_AREA] < 8:
            return None, 0
        symbol = normalized_mask((labels == component).astype('uint8')*255)
        is_red = np.count_nonzero(red & (labels == component)) > stats[component, cv2.CC_STAT_AREA] / 2
        scores = []
        for suit in ('dh' if is_red else 'cs'):
            template = self.templates[suit]
            score = np.count_nonzero(symbol & template) / max(1, np.count_nonzero(symbol | template))
            scores.append((score, suit))
        scores.sort(reverse=True)
        best, suit = scores[0]
        if best < .65 or best - scores[1][0] < .08:
            return None, best
        return suit, best


class SkinGlyphMatcher:
    """Skin calibration from isolated glyphs, never whole-frame state lookup."""
    def __init__(self, video=False):
        self.templates={}
        for path in (Path(__file__).parent/'vision_templates').glob('*.png'):
            if path.stem.endswith('_video') and not video:
                continue
            self.templates[path.stem]=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)

    def read(self, image, kind):
        if not image.size: return None,0
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if float(gray.max())-float(gray.min())<20: return None,0
        _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        clean=np.zeros_like(mask)
        for i,(x,y,w,h,area) in enumerate(stats[1:],1):
            if area<3: continue
            # Exclude a card outline crossing the complete crop height.
            if (x==0 or x+w==mask.shape[1]) and h>=mask.shape[0]*.85: continue
            clean[labels==i]=255
        if cv2.countNonZero(clean)<8: return None,0
        if kind=='suit':
            _,components,regions,_=cv2.connectedComponentsWithStats(clean)
            candidates=[(i,r) for i,r in enumerate(regions[1:],1) if r[4]>=5]
            if not candidates: return None,0
            main,rect=max(candidates,key=lambda pair:pair[1][4])
            x,y,w,h,_=rect
            selected=np.zeros_like(clean)
            for i,(xx,yy,ww,hh,area) in candidates:
                if i==main or (xx>=x and xx+ww<=x+w and yy<=y+h+4 and yy+hh>=y):
                    selected[components==i]=255
            clean=selected
        scored=[]
        for key,template in self.templates.items():
            if not key.startswith(kind+'_'): continue
            best=0
            for angle in (-12,-6,0,6,12):
                padded=cv2.copyMakeBorder(clean,12,12,12,12,cv2.BORDER_CONSTANT,value=0)
                matrix=cv2.getRotationMatrix2D((padded.shape[1]/2,padded.shape[0]/2),angle,1)
                rotated=cv2.warpAffine(padded,matrix,(padded.shape[1],padded.shape[0]))
                symbol=normalized_mask(rotated)
                score=np.count_nonzero((symbol>100)&(template>100))/max(1,np.count_nonzero((symbol>100)|(template>100)))
                best=max(best,score)
            scored.append((best,key.split('_')[1]))
        by_label={}
        for score,label in scored:
            by_label[label]=max(score,by_label.get(label,0))
        scored=sorted(((score,label) for label,score in by_label.items()),reverse=True)
        if not scored: return None,0
        score,label=scored[0]
        if score<.73 or (len(scored)>1 and score-scored[1][0]<.09): return None,score
        return label,score


def locate_board(image):
    """Locate visible card faces within the felt; tolerate 3/4/5-card spacing."""
    region=image[210:380,280:740]
    gray=cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
    _,mask=cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if 38<=w<=80 and 65<=h<=105 and .48<w/h<.9 and cv2.contourArea(c)/(w*h)>.65:
            boxes.append((x+280,y+210,w,h))
    boxes.sort()
    if len(boxes) not in (3,4,5): return []
    if max(b[1] for b in boxes)-min(b[1] for b in boxes)>8: return []
    return boxes


class TableReader:
    # Reference images: expanded 1024x765; compact 1024x677.
    # Each seat is (name rect, stack rect, status rect, bet rect, card area).
    def __init__(self, layout='auto', confirmations=2, viewport=None):
        self.layout = layout
        self.confirmations = confirmations
        if confirmations < 1:
            raise ValueError('confirmations must be positive')
        self.history = deque(maxlen=confirmations)
        self.suits = SuitMatcher()
        self.skin = SkinGlyphMatcher(video=self.layout=='video')
        self.viewport = viewport

    def card(self, image, bbox, suit_box=None):
        x,y,w,h = bbox
        card = image[y:y+h,x:x+w]
        if card.size == 0 or float(card.max())-float(card.min()) < 25:
            return Reading(bbox=bbox)
        # Rank and small suit occupy the left index. Ten is explicitly supported.
        rh, rw = max(1,int(h*.43)), max(1,int(w*.48))
        rank_img = card[2:rh,3:rw]
        gray = cv2.cvtColor(cv2.resize(rank_img,None,fx=4,fy=4),cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        gray = cv2.copyMakeBorder(gray,10,10,10,10,cv2.BORDER_CONSTANT,value=255)
        raw = pytesseract.image_to_string(gray,config='--psm 7 -c tessedit_char_whitelist=A2345678910TJQK').strip()
        rank = 'T' if raw == '10' else raw
        skin_rank,rank_conf=self.skin.read(rank_img,'rank')
        if skin_rank: rank=skin_rank
        suit_img=card[int(h*.43):int(h*.72),3:rw]
        if suit_box:
            sx,sy,sw,sh=suit_box
            suit_img=image[sy:sy+sh,sx:sx+sw]
        suit, confidence = self.suits.read(suit_img)
        skin_suit,skin_conf=self.skin.read(suit_img,'suit')
        if skin_suit: suit,confidence=skin_suit,skin_conf
        value = rank+suit if rank in '23456789TJQKA' and len(rank)==1 and suit else None
        return Reading(value, min(confidence,.85) if value else 0,raw,bbox)

    def read(self, frame):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected a nonempty BGR image')
        if self.viewport:
            x,y,w,h=self.viewport
            if min(x,y)<0 or min(w,h)<=0 or x+w>frame.shape[1] or y+h>frame.shape[0]:
                raise ValueError('Viewport falls outside the frame')
            frame=frame[y:y+h,x:x+w]
        video = self.layout=='video'
        compact = video or self.layout=='compact' or (self.layout=='auto' and frame.shape[0]/frame.shape[1] < .70)
        height=684 if video else (677 if compact else 765)
        image=cv2.resize(frame,(1024,height))
        if compact:
            names=[(130,124),(445,84),(826,126),(873,261),(800,434),(509,502),(163,433),(76,262)]
            stacks=[(163,151),(466,109),(848,151),(902,289),(832,462),(535,530),(184,461),(97,289)]
            cards=[(155,80),(500,45),(800,80),(880,215),(760,390),(445,449),(176,390),(100,212)]
            board_y,pot_y=265,230
        else:
            names=[(119,150),(454,105),(859,150),(908,300),(830,490),(514,562),(150,490),(50,300)]
            stacks=[(143,179),(474,133),(874,180),(929,332),(849,522),(547,594),(158,522),(76,331)]
            cards=[(155,100),(500,55),(800,100),(890,245),(760,437),(489,481),(155,429),(90,245)]
            board_y,pot_y=292,265
        if video:
            stacks=[(153,154),(466,113),(845,155),(904,291),(813,464),(532,527),(186,462),(97,289)]
        fields={}
        avatars = ([(250,150),(564,111),(784,151),(839,287),(764,461),(470,525),(270,461),(195,288)]
                   if compact else [(240,178),(578,130),(814,178),(876,325),(793,517),(479,590),(262,515),(178,326)])
        def read(key,box,numeric=False):
            fields[key]=text_read(image,box,numeric)
        fields['pot']=amount_read(image,(521 if compact else 530,pot_y,53,24))
        if video: fields['pot']=chip_amount(image,(470,230,120,25))
        read('blinds',(898,25 if not compact else 9,90,40))
        name_boxes = ([(140,129,70,18),(455,85,73,20),(830,129,70,18),(875,265,84,19),(803,438,78,19),(511,501,81,20),(175,437,51,19),(78,264,84,19)]
                      if compact else [(122,151,78,21),(459,106,79,21),(866,151,75,21),(914,300,89,22),(833,490,86,21),(518,562,81,22),(151,490,65,21),(51,301,91,22)])
        # Explicit inward bet regions; interpolation crosses unrelated cards and labels.
        bets = ([(296,188),(474,188),(674,188),(746,286),(672,378),(480,407),(297,375),(220,286)]
                if compact else [(300,224),(476,218),(678,224),(758,326),(677,422),(480,452),(297,418),(220,326)])
        if video:
            bets=[(288,191),(490,191),(666,191),(734,278),(670,377),(488,378),(307,377),(222,278)]
            name_boxes=[(139,129,76,21),(448,88,77,22),(835,130,67,22),(871,267,85,22),(802,437,80,22),(508,503,82,23),(163,437,74,23),(76,266,84,23)]
        for i,((nx,ny),(sx,sy),(cx,cy)) in enumerate(zip(names,stacks,cards)):
            prefix=f'seat_{i}'
            read(prefix+'.name',name_boxes[i])
            fields[prefix+'.stack']=amount_read(image,(sx-4 if compact and i==4 else sx,sy,70,23))
            if video:
                stack_search=[(130,153,90,25),(444,112,92,25),(815,155,85,25),(880,290,82,25),
                              (793,462,92,25),(500,525,90,25),(163,460,74,25),(75,290,84,25)][i]
                fields[prefix+'.stack']=chip_amount(image,stack_search)
            # Only known label positions are searched; avatar text is separate.
            status_box=(155,179,48,17) if compact and i==0 else ((829,549,86,16) if not compact and i==4 else (nx,ny+48,92,22))
            if video:
                status_box=[(143,181,65,19),(454,140,67,20),(819,181,68,20),(873,317,66,22),(802,486,78,21),(505,551,70,22),(163,488,68,22),(76,315,74,22)][i]
            read(prefix+'.status',status_box)
            ax,ay=avatars[i]
            read(prefix+'.avatar_text',(ax-31,ay-13,62,28))
            fields[prefix+'.turn_ring']=turn_ring(image,(ax,ay))
            bx,by=bets[i]
            fields[prefix+'.bet']=amount_read(image,(bx,by,64,24))
            for j in range(2):
                box=(cx+j*33,cy,34,64)
                if not compact and i==5:
                    box=(489+37*j,486-3*j,64,91)
                if not compact and i==6:
                    box=(155+32*j,429-2*j,64,70)
                suit_box=None
                if not compact and i==5:
                    suit_box=(496,527,29,27) if j==0 else (526,524,29,25)
                if not compact and i==6:
                    suit_box=(162,462,22,23) if j==0 else (185,460,23,23)
                if video and i==5:
                    box=(480+35*j,432-2*j,61,80)
                    suit_box=(487,468,25,25) if j==0 else (514,465,25,25)
                fields[prefix+f'.card_{j}']=self.card(image,box,suit_box)
        board_boxes=locate_board(image) if video else []
        for i in range(5):
            if video:
                fields[f'board_{i}']=self.card(image,board_boxes[i]) if i<len(board_boxes) else Reading()
            else:
                fields[f'board_{i}']=self.card(image,(361+i*65,board_y,64,91))
        # Dealer markers are bright, approximately round connected components.
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        dealers=[]
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            area=cv2.contourArea(contour)
            if 19<=w<=38 and 19<=h<=38 and .8<w/h<1.2 and area/(w*h)>.6:
                roi=image[y+int(h*.2):y+int(h*.8),x+int(w*.2):x+int(w*.8)]
                roi=cv2.copyMakeBorder(roi,6,6,6,6,cv2.BORDER_CONSTANT,value=(30,30,30))
                raw=pytesseract.image_to_string(cv2.resize(roi,None,fx=4,fy=4),config='--psm 10 -c tessedit_char_whitelist=D').strip()
                if raw=='D':
                    nearest=min(range(8),key=lambda i:(avatars[i][0]-x-w/2)**2+(avatars[i][1]-y-h/2)**2)
                    dealers.append((nearest,(x,y,w,h)))
        fields['dealer_seat']=Reading(dealers[0][0],.8,'D',dealers[0][1]) if len(dealers)==1 else Reading()
        # Reject physically impossible duplicate identified cards.

        groups={}
        warnings=['Layout is skin-specific; verify overlay and use --layout/--viewport when needed.']
        for key,r in fields.items():
            if ('card_' in key or key.startswith('board_')) and r.value:
                groups.setdefault(r.value,[]).append(key)
        for keys in groups.values():
            if len(keys)>1:
                for key in keys:
                    fields[key].candidate=fields[key].value
                    fields[key].value=None
                    fields[key].confidence=0
                warnings.append('Duplicate card observations rejected at '+', '.join(keys)+'; candidates retained')
        values={k:r.value for k,r in fields.items()}
        self.history.append(values)
        confirmed={k:v for k,v in values.items() if v is not None and len(self.history)==self.confirmations and all(h.get(k)==v for h in self.history)}
        return {'timestamp':time.time(),'layout':'video' if video else ('compact' if compact else 'expanded'),
                'reference_size':[1024,height],'fields':{k:asdict(r) for k,r in fields.items()},
                'confirmed':confirmed,'table':summarize(fields),'warnings':warnings}


def annotate(frame,result):
    out=cv2.resize(frame,tuple(result['reference_size']))
    for key,r in result['fields'].items():
        if not r['bbox']: continue
        x,y,w,h=r['bbox']
        color=(0,200,0) if r['value'] is not None else (0,160,255)
        cv2.rectangle(out,(x,y),(x+w,y+h),color,1)
        if r['value'] is not None:
            cv2.putText(out,str(r['value']),(x,y-3),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image',type=Path)
    source.add_argument('--video',type=Path)
    source.add_argument('--live',action='store_true')
    p.add_argument('--layout',choices=['auto','compact','expanded','video'],default='auto')
    p.add_argument('--viewport',type=int,nargs=4,metavar=('X','Y','WIDTH','HEIGHT'))
    p.add_argument('--confirmations',type=int,default=2)
    p.add_argument('--overlay',type=Path,help='Write last-frame recognition overlay')
    p.add_argument('--interval',type=float,default=.5)
    args=p.parse_args()
    if args.interval<0:
        p.error('--interval must be nonnegative')
    reader=TableReader(args.layout,args.confirmations,args.viewport)
    def process(frame, source_seconds=None, frame_index=None):
        result=reader.read(frame)
        result['source_seconds']=source_seconds
        result['frame_index']=frame_index
        print(json.dumps(result),flush=True)
        if args.overlay:
            cropped=frame
            if args.viewport:
                x,y,w,h=args.viewport
                cropped=frame[y:y+h,x:x+w]
            if not cv2.imwrite(str(args.overlay),annotate(cropped,result)):
                raise RuntimeError('Could not write overlay')
    if args.image:
        frame=cv2.imread(str(args.image))
        if frame is None:
            p.error('Cannot decode input image')
        process(frame)
    elif args.video:
        cap=cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            p.error('Cannot open video')
        try:
            fps=cap.get(cv2.CAP_PROP_FPS) or 30
            stride=max(1,round(fps*args.interval))
            index=0
            while True:
                ok,frame=cap.read()
                if not ok: break
                if index%stride==0: process(frame,cap.get(cv2.CAP_PROP_POS_MSEC)/1000,index)
                index+=1
        finally:
            cap.release()
    else:
        import mss
        with mss.mss() as sct:
            try:
                while True:
                    process(cv2.cvtColor(np.asarray(sct.grab(sct.monitors[1])),cv2.COLOR_BGRA2BGR))
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                pass


if __name__=='__main__':
    main()
