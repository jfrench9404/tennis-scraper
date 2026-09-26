"""Exercise CLI plumbing without downloading or running the unrelated YOLO models."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tennis_vision.tracking import Detection


@unittest.skipUnless(importlib.util.find_spec('yaml') and importlib.util.find_spec('tqdm'),'CLI optional dependencies unavailable')
class TemporalCliTest(unittest.TestCase):
    def test_preserves_tail_alignment_and_never_silently_uses_yolo_ball(self):
        fake_ultralytics=types.ModuleType('ultralytics')
        fake_ultralytics.YOLO=lambda path: object()
        with patch.dict(sys.modules,{'ultralytics':fake_ultralytics}):
            cli=importlib.import_module('tennis_vision.cli')
        class Temporal:
            def predict(self,frames):
                return [(Detection('ball',(10,10,14,14),.8,source='gridtracknet') if i!=4 else None,{}) for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); video=root/'source.mp4'; weights=root/'test-model.onnx'
            weights.write_bytes(b'unit-test-model-placeholder')
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,48))
            if not writer.isOpened(): self.skipTest('MP4 encoder unavailable')
            for i in range(7): writer.write(np.full((48,64,3),i,np.uint8))
            writer.release()
            argv=['tennis_vision','--input',str(video),'--output',str(root/'result'),
                  '--no-auto-court','--no-detail-pass','--no-progress','--temporal-ball','--ball-model',str(weights)]
            with patch.object(sys,'argv',argv), patch.object(cli,'GridTrackNetDetector',return_value=Temporal()), \
                 patch.object(cli,'YOLO',return_value=object()), \
                 patch.object(cli,'detect',return_value=[Detection('ball',(30,30,34,34),.99)]), \
                 patch.object(cli,'detect_pose',return_value=[]):
                cli.main()
            rows=[json.loads(line) for line in (root/'result/events.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame'] for r in rows],list(range(7)))
            self.assertEqual(rows[4]['tracks'],[])
            self.assertTrue(all(t['source']=='gridtracknet' for r in rows for t in r['tracks']))
            cap=cv2.VideoCapture(str(root/'result/annotated.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),7)
            cap.release()
            self.assertTrue((root/'result/ball-model.json').exists())
