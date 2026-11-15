"""Locate the known game client on a desktop using its static rail/UI features.

This is a conservative match to the retained layout, not a universal window finder.
"""
from pathlib import Path
import cv2
import numpy as np


class TableLocator:
    def __init__(self):
        self.orb=cv2.ORB_create(nfeatures=3500,fastThreshold=12)
        self.references=[]
        for name in ('video-40.png','video-70.png'):
            image=cv2.imread(str(Path(__file__).parent/'fixtures'/name))
            if image is None: continue
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            mask=np.zeros(gray.shape,np.uint8)
            for x,y,w,h in [(100,130,825,65),(180,435,670,50),(80,180,105,260),(840,180,105,260),(0,0,85,75),(850,0,174,75),(40,610,180,74)]:
                mask[y:y+h,x:x+w]=255
            points,desc=self.orb.detectAndCompute(gray,mask)
            if desc is not None: self.references.append((points,desc,gray.shape))

    def locate(self,image):
        scale=min(1,1920/image.shape[1])
        small=cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        scene_points,scene_desc=self.orb.detectAndCompute(gray,None)
        if scene_desc is None: return None
        best=None
        for points,desc,(height,width) in self.references:
            pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc,scene_desc,k=2)
            good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.7*p[1].distance]
            if len(good)<18: continue
            source=np.float32([points[m.queryIdx].pt for m in good])
            target=np.float32([scene_points[m.trainIdx].pt for m in good])
            matrix,inliers=cv2.findHomography(source,target,cv2.RANSAC,4)
            if matrix is None or inliers is None: continue
            accepted=source[inliers.ravel().astype(bool)]
            count=len(accepted)
            if count<18 or count/len(good)<.45: continue
            if np.ptp(accepted[:,0])<500 or np.ptp(accepted[:,1])<220: continue
            corners=cv2.perspectiveTransform(np.float32([[[0,0],[width,0],[width,height],[0,height]]]),matrix)[0]
            if not np.isfinite(corners).all(): continue
            left,top=corners.min(axis=0);right,bottom=corners.max(axis=0)
            w,h=right-left,bottom-top
            if w<300 or h<200 or w*h<small.shape[0]*small.shape[1]*.18: continue
            if not 1.35<w/h<1.65: continue
            # Desktop game clients have axis-aligned rectangular edges.
            if abs(corners[0,1]-corners[1,1])>h*.025 or abs(corners[0,0]-corners[3,0])>w*.025: continue
            if left<-6 or top<-6 or right>small.shape[1]+6 or bottom>small.shape[0]+6: continue
            x=max(0,round(left/scale));y=max(0,round(top/scale))
            endx=min(image.shape[1],round(right/scale));endy=min(image.shape[0],round(bottom/scale))
            if best is None or count>best[0]: best=(count,[x,y,endx-x,endy-y])
        return best[1] if best else None
