These are isolated glyph masks calibrated from the expanded user-supplied image
in ../fixtures. They are not full-card or full-screen matching images.

Sources (x, y, width, height) in the 1024x765 frame:
- rank_J: (368,294,19,38)
- rank_9: (432,294,20,38)
- rank_T: (495,294,27,38)
- rank_2: (561,294,22,38)
- rank_3: (495,490,25,37)
- rank_7: (529,487,25,37)
- suit_c: (366,333,22,23)
- suit_h: (495,333,23,23)
- suit_c_dim: (496,527,29,27)
- suit_s: (164,464,17,18)

Masks use grayscale Otsu inverse thresholding, then normalized_mask() from
../table_reader.py. Unrepresented ranks use OCR and diamonds use the font
fallback. Tests on these frames are calibration regressions, not held-out
accuracy. Brightness/resizing checks are synthetic robustness checks only.

Additional calibration from the user-supplied recording (normalized 1024x684):
- suit_s_video: frame 40s, (367,295,18,20)
- suit_c_video: frame 40s, (428,295,18,20)
- rank_Q_video: frame 50s, (607,260,24,33)
These are calibration samples, not independent test data. Diamonds in the clip
continue to use the existing font matcher. Multiple templates of a label are
combined by taking the strongest match before comparing different labels.
Templates ending `_video` are loaded only for the explicit video layout. This
keeps calibration for one rendering from introducing suit ambiguity in the other.
