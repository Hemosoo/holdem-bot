import unittest
from pathlib import Path
import cv2
import numpy as np
from table_locator import TableLocator


class LocatorTests(unittest.TestCase):
    def test_left_offset_resizing_and_changed_hand(self):
        locator=TableLocator()
        for name in ('video-40.png','video-70.png'):
            image=cv2.imread(str(Path(__file__).parent/'fixtures'/name))
            for factor in (.7,1,1.3):
                game=cv2.resize(image,None,fx=factor,fy=factor)
                h,w=game.shape[:2]
                desktop=np.full((h+80,w+480,3),30,np.uint8)
                desktop[30:30+h,10:10+w]=game
                crop=locator.locate(desktop)
                self.assertIsNotNone(crop)
                self.assertTrue(all(abs(a-b)<=5 for a,b in zip(crop,[10,30,w,h])),(name,factor,crop))

    def test_does_not_match_unrelated_screen(self):
        self.assertIsNone(TableLocator().locate(np.zeros((900,1600,3),np.uint8)))


if __name__=='__main__': unittest.main()
