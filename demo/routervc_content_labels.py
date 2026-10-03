"""Sparse, offline OCR/face-landmark proxy labels; never receiver inputs.

OpenCV CPU only. Source detection establishes fixed comparison regions at
frames 0/8/16. OCR agreement is not semantic truth; YuNet is not an identity
recognizer. No detection/low confidence/missing face stays unknown, never safe.
This bounded provider does not route, train models, or transmit labels/masks.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

COMMIT = '47534e27c9851bb1128ccc0102f1145e27f23f98'
OFFICIAL = 'https://raw.githubusercontent.com/opencv/opencv_zoo/' + COMMIT
MEDIA = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/' + COMMIT
ASSETS = Path('/root/autodl-tmp/DCVC/models/routervc-content-opencv')
CHARSET_URL = ('https://raw.githubusercontent.com/opencv/opencv_zoo/'
               'aab69020085e9b6390723b61f9789ec56b96b07e/models/text_recognition_crnn/charset_3944_CN.txt')
CHARSET_SHA256 = '8027c9832d86764feccd9bdd8974829c86994617e5787f178ed97db2bda1481a'
STATES = ('B', 'E', 'G', 'EG')
FRAMES = (0, 8, 16)
SCHEMA = 'routervc-sparse-content-labels-v1'
MODELS = {
    'text_detector': dict(folder='text_detection_ppocr', file='text_detection_cn_ppocrv3_2023may.onnx',
                         bytes=2423490, sha256='03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587', license='Apache-2.0'),
    'text_recognizer': dict(folder='text_recognition_crnn', file='text_recognition_CRNN_CN_2021nov.onnx',
                           bytes=72807160, sha256='c760bf82d684b87dfabb288e6c0f92d41a8cd6c1780661ca2c3cd10c2065a9ba', license='Apache-2.0'),
    'face_detector': dict(folder='face_detection_yunet', file='face_detection_yunet_2023mar.onnx',
                         bytes=232589, sha256='8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4', license='MIT'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def output_lock(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another content-label process owns this output') from None
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def resource_record():
    disks={}
    for path in ('/root','/root/autodl-tmp','/root/autodl-fs'):
        stat=os.statvfs(path)
        disks[path]=dict(total_bytes=stat.f_blocks*stat.f_frsize,
                         used_bytes=(stat.f_blocks-stat.f_bfree)*stat.f_frsize,
                         available_bytes=stat.f_bavail*stat.f_frsize)
    return dict(peak_CPU_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                GPU_used=False,disks=disks)


def clean_download_environment(environment):
    """Clean this subprocess only; existing training and proxy transport survive."""
    removed = []
    explicit = {'HF_ENDPOINT', 'HF_HUB_ENDPOINT', 'HUGGINGFACE_HUB_BASE_URL',
                'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'UV_INDEX_URL', 'UV_EXTRA_INDEX_URL'}
    clean = dict(environment)
    for key in list(clean):
        upper = key.upper()
        if upper in explicit or 'MIRROR' in upper:
            removed.append(key); del clean[key]
    return clean, sorted(removed)


def fetch_assets(root):
    """Resumable official downloads; model content verified against Git LFS SHA."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    environment, removed = clean_download_environment(os.environ)
    assets = []
    for key, spec in MODELS.items():
        assets.append(dict(name=spec['file'], url=f"{MEDIA}/models/{spec['folder']}/{spec['file']}",
                           expected_sha256=spec['sha256'], expected_bytes=spec['bytes'],
                           license=spec['license'], role=key))
        for name in ('LICENSE', 'README.md'):
            assets.append(dict(name=f"{key}_{name}", url=f"{OFFICIAL}/models/{spec['folder']}/{name}"))
    assets.append(dict(name='official_crnn.py', url=f'{OFFICIAL}/models/text_recognition_crnn/crnn.py'))
    assets.append(dict(name='charset_3944_CN.txt',url=CHARSET_URL,expected_sha256=CHARSET_SHA256,
                       expected_bytes=15548,license='Apache-2.0',role='original_CN_model_charset'))
    records = []
    for item in assets:
        path = root/item['name']
        if not path.exists():
            partial = path.with_name(path.name+'.part')
            subprocess.run(['curl', '--fail', '--location', '--continue-at', '-', '--retry', '3',
                            '--connect-timeout', '20', '--max-time', '1800', '--output', str(partial),
                            item['url']], env=environment, check=True)
            if item.get('expected_bytes') is not None and partial.stat().st_size != item['expected_bytes']:
                raise ValueError(f'wrong model size: {partial}')
            if item.get('expected_sha256') and digest(partial) != item['expected_sha256']:
                raise ValueError(f'wrong model SHA256: {partial}')
            os.replace(partial, path)
        if item.get('expected_bytes') is not None and path.stat().st_size != item['expected_bytes']:
            raise ValueError(f'existing model size differs: {path}')
        if item.get('expected_sha256') and digest(path) != item['expected_sha256']:
            raise ValueError(f'existing model hash differs: {path}')
        records.append(dict(item, path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path)))
        print(json.dumps(dict(asset=item['name'], verified=True, bytes=path.stat().st_size)), flush=True)
    record = dict(complete=True, upstream='opencv/opencv_zoo', commit=COMMIT,
                  mirror_environment_names_removed=removed, assets=records)
    target = root/'assets.json'
    if target.exists():
        old = json.loads(target.read_text())
        old_items = {item['name']:item for item in old['assets']}
        new_items = {item['name']:item for item in record['assets']}
        if any(new_items.get(k)!=v for k,v in old_items.items()) or old['commit'] != COMMIT:
            raise ValueError('asset provenance changed; use a new asset directory')
        if old_items != new_items:
            save(root/'assets_before_charset_addition.json',old)
            save(target,record)
    else:
        save(target, record)
    return record


def charset_from_official(path):
    """Read the official character lines, preserving its literal backslash entry.

    The upstream Python triple-quoted string accidentally treats backslash plus
    newline as an escape and loses a character. Reading the source token as data
    preserves the advertised 3944-character list and avoids shifted indices.
    """
    source = Path(path).read_text()
    if Path(path).name == 'charset_3944_CN.txt':
        if digest(path) != CHARSET_SHA256:
            raise ValueError('original model charset SHA mismatch')
        text = ''.join(source.splitlines())
        if len(text) != 3944:
            raise ValueError('original model charset length mismatch')
        return text
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHARSET_CN_3944'
                                                for t in node.targets):
            token = ast.get_source_segment(source,node.value)
            if not token.startswith("'''") or not token.endswith("'''"):
                raise ValueError('unexpected official charset literal')
            text = ''.join(token[3:-3].splitlines())
            if len(text) != 3944 or not set('0123456789Aa中').issubset(text):
                raise ValueError('unexpected official Chinese charset')
            return text
    raise ValueError('official Chinese charset missing')


def ctc_decode(output, charset):
    """Greedy CTC and uncalibrated emitted-character confidence, excluding blanks."""
    import numpy as np
    logits = np.asarray(output, dtype=np.float64)
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits[:, 0, :]
    # CN's official model has 3946 outputs (blank + 3944 listed symbols + one
    # unlisted class). Never invent its meaning or renormalize it away.
    if logits.ndim != 2 or logits.shape[1] not in (len(charset)+1,len(charset)+2) or not np.isfinite(logits).all():
        raise ValueError('unexpected CRNN output/charset')
    if np.all(logits >= 0) and np.all(logits <= 1) and np.allclose(logits.sum(1), 1, atol=1e-4):
        probability = logits
    else:
        probability = np.exp(logits-logits.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
    indices = probability.argmax(1)
    if np.any(indices > len(charset)):
        return dict(text='',confidence=0.,reason='unmapped_upstream_output_class',
                    unmapped_class_indices=sorted(set(int(i) for i in indices if i>len(charset))),
                    confidence_scope='unknown_not_a_character')
    characters, scores = [], []
    previous = 0
    for row, index in zip(probability, indices):
        if index and index != previous:
            characters.append(charset[index-1]); scores.append(float(row[index]))
        elif index and scores:
            scores[-1] = max(scores[-1], float(row[index]))
        previous = index
    confidence = math.exp(sum(math.log(max(v, 1e-12)) for v in scores)/len(scores)) if scores else 0.
    return dict(text=''.join(characters), confidence=confidence,
                confidence_scope='uncalibrated_geometric_mean_emitted_character_probability')


def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        next_row = [i]
        for j, cb in enumerate(b, 1):
            next_row.append(min(next_row[-1]+1, row[j]+1, row[j-1]+(ca != cb)))
        row = next_row
    return row[-1]


def compare_text(reference, candidate, threshold=.75):
    reference_known = bool(reference['text']) and reference['confidence'] >= threshold
    candidate_known = bool(candidate['text']) and candidate['confidence'] >= threshold
    known = reference_known and candidate_known
    digits = lambda text: ''.join(c for c in text if c in '0123456789')
    return dict(status='known' if known else 'unknown',
                normalized_edit_error=(edit_distance(reference['text'], candidate['text'])/
                                       max(len(reference['text']), len(candidate['text']))) if known else None,
                source_digits=digits(reference['text']) if reference_known else None,
                candidate_digits=digits(candidate['text']) if candidate_known else None,
                digits_changed=(digits(reference['text']) != digits(candidate['text'])) if known else None,
                readability_loss_proxy=(not candidate_known) if reference_known else None,
                readability_scope='OCR_confidence_or_empty_output_not_human_readability',
                candidate=candidate)


def box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    intersect = max(0., min(ax+aw, bx+bw)-max(ax, bx))*max(0., min(ay+ah, by+bh)-max(ay, by))
    union = aw*ah+bw*bh-intersect
    return intersect/union if union > 0 else 0.


def match_face(faces, reference_box, min_iou=.3):
    ranked = sorted(((box_iou(face['box'], reference_box), face) for face in faces),
                    key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] < min_iou:
        return None, 'missing_or_low_overlap'
    if len(ranked) > 1 and ranked[1][0] >= min_iou and ranked[0][0]-ranked[1][0] < .1:
        return None, 'ambiguous_multiple_faces'
    return ranked[0][1], 'matched_by_source_geometry_not_identity'


def compare_face(reference, candidate):
    import numpy as np
    if reference is None or candidate is None:
        return dict(status='unknown', normalized_landmark_error=None, face_identity_assessed=False)
    diagonal = math.hypot(reference['box'][2], reference['box'][3])
    if diagonal <= 0:
        raise ValueError('empty source face')
    a, b = np.asarray(reference['landmarks']), np.asarray(candidate['landmarks'])
    if a.shape != (5, 2) or b.shape != (5, 2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('five finite landmarks required')
    return dict(status='known', normalized_landmark_error=float(np.linalg.norm(a-b, axis=1).mean()/diagonal),
                face_identity_assessed=False)


class OpenCVProvider:
    def __init__(self, assets=ASSETS, text_threshold=.75, face_threshold=.9):
        import cv2
        import numpy as np
        self.cv, self.np = cv2, np
        self.assets = Path(assets)
        self.provenance = json.loads((self.assets/'assets.json').read_text())
        if self.provenance['commit'] != COMMIT:
            raise ValueError('provider assets must use pinned upstream commit')
        for artifact in self.provenance['assets']:
            if digest(self.assets/artifact['name']) != artifact['sha256']:
                raise ValueError('provider asset changed: '+artifact['name'])
        for spec in MODELS.values():
            if digest(self.assets/spec['file']) != spec['sha256']:
                raise ValueError('model is not the pinned upstream model')
        cv2.setNumThreads(2)
        self.text_threshold, self.face_threshold = text_threshold, face_threshold
        self.charset = charset_from_official(self.assets/'charset_3944_CN.txt')
        self.text = cv2.dnn_TextDetectionModel_DB(str(self.assets/MODELS['text_detector']['file']))
        self.text.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.text.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.text.setBinaryThreshold(.3); self.text.setPolygonThreshold(.6)
        self.text.setUnclipRatio(2.); self.text.setMaxCandidates(200)
        self.text.setInputMean((123.675, 116.28, 103.53))
        self.text.setInputScale(1/255/np.asarray([.229, .224, .225]))
        self.recognizer = cv2.dnn.readNet(str(self.assets/MODELS['text_recognizer']['file']))
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.face = cv2.FaceDetectorYN.create(str(self.assets/MODELS['face_detector']['file']), '',
                                            (320, 320), score_threshold=face_threshold,
                                            nms_threshold=.3, top_k=5000,
                                            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                                            target_id=cv2.dnn.DNN_TARGET_CPU)

    def recognize(self, bgr, polygon):
        cv, np = self.cv, self.np
        points = np.asarray(polygon, np.float32)
        # DB gives four vertices; reconstruct BL,TL,TR,BR explicitly.
        total, difference = points.sum(1), points[:, 1]-points[:, 0]
        ordered = np.asarray([points[difference.argmax()], points[total.argmin()],
                              points[difference.argmin()], points[total.argmax()]], np.float32)
        if len(np.unique(ordered, axis=0)) != 4:
            return dict(text='', confidence=0., reason='degenerate_text_quad')
        target = np.asarray([[0,31], [0,0], [99,0], [99,31]], np.float32)
        patch = cv.warpPerspective(bgr, cv.getPerspectiveTransform(ordered, target), (100,32))
        blob = cv.dnn.blobFromImage(patch, scalefactor=1/127.5, size=(100,32), mean=(127.5,)*3)
        self.recognizer.setInput(blob)
        return ctc_decode(self.recognizer.forward(), self.charset)

    def detect_text(self, bgr):
        cv, np = self.cv, self.np
        height, width = bgr.shape[:2]
        scale = min(1., 736/max(height, width))
        w, h = max(1, round(width*scale)), max(1, round(height*scale))
        resized = cv.resize(bgr, (w,h))
        padded = cv.copyMakeBorder(resized, 0, (-h)%32, 0, (-w)%32, cv.BORDER_CONSTANT, value=(0,0,0))
        self.text.setInputSize((padded.shape[1], padded.shape[0]))
        boxes, scores = self.text.detect(padded)
        records = []
        for polygon, confidence in zip(boxes, scores):
            points = np.asarray(polygon, dtype=np.float64)
            points[:, 0] *= width/w; points[:, 1] *= height/h
            points[:, 0] = points[:, 0].clip(0, width-1)
            points[:, 1] = points[:, 1].clip(0, height-1)
            x, y = points.min(0); x2, y2 = points.max(0)
            recognized = self.recognize(bgr, points)
            reliable = (float(confidence) >= .6 and min(x2-x, y2-y) >= 8 and
                        recognized['confidence'] >= self.text_threshold and bool(recognized['text']))
            records.append(dict(polygon=points.tolist(), box=[float(x),float(y),float(x2-x),float(y2-y)],
                                detector_confidence=float(confidence), source_ocr=recognized,
                                status='present' if reliable else 'unknown', importance=1. if reliable else None,
                                importance_scope='text_presence_proxy_not_general_importance_ground_truth',
                                source_reference_verified=False,reference_verification='automatic_OCR_only',
                                candidate_measurements_verified=False,
                                character_errors_trainable=False))
        return records

    def _detect_faces_at_scale(self,bgr,scale):
        height, width = bgr.shape[:2]
        if height < 24 or width < 24:
            return []
        w,h=max(1,round(width*scale)),max(1,round(height*scale))
        image=bgr if scale==1. else self.cv.resize(bgr,(w,h))
        self.face.setInputSize((w,h))
        _, faces = self.face.detect(image)
        if faces is None:
            return []
        records=[reproject_face(row,width/w,height/h,scale) for row in faces if row[-1]>=self.face_threshold]
        return [r for r in records if min(r['box'][2:4])>=24]

    def detect_faces(self,bgr,forced_scale=None):
        if forced_scale is not None:
            return self._detect_faces_at_scale(bgr,forced_scale)
        faces=self._detect_faces_at_scale(bgr,1.)
        return faces if faces else self._detect_faces_at_scale(bgr,.5)

    def frame(self, source_rgb, candidates_rgb):
        cv = self.cv
        source = cv.cvtColor(source_rgb, cv.COLOR_RGB2BGR)
        candidates = {key: cv.cvtColor(value, cv.COLOR_RGB2BGR) for key, value in candidates_rgb.items()}
        texts = self.detect_text(source)
        for text in texts:
            text['states'] = {key: compare_text(text['source_ocr'], self.recognize(value, text['polygon']),
                                                self.text_threshold) for key, value in candidates.items()}
            for state in text['states'].values():
                state.update(error_scope='diagnostic_agreement_with_unverified_source_OCR',
                             character_error_trainable=False)
            if text['status'] == 'unknown':
                for state in text['states'].values():
                    state.update(status='unknown', normalized_edit_error=None, digits_changed=None,
                                 readability_loss_proxy=None)
        faces = []
        for detected in self.detect_faces(source):
            x,y,w,h = detected['box']
            x0,y0 = max(0, int(x-w*.25)), max(0, int(y-h*.25))
            x1,y1 = min(source.shape[1], math.ceil(x+w*1.25)), min(source.shape[0], math.ceil(y+h*1.25))
            local_reference_box = [x-x0,y-y0,w,h]
            scale=detected['detection_scale']
            reference, reason = match_face(self.detect_faces(source[y0:y1,x0:x1],forced_scale=scale), local_reference_box)
            states = {}
            for key, value in candidates.items():
                candidate, why = match_face(self.detect_faces(value[y0:y1,x0:x1],forced_scale=scale), local_reference_box)
                states[key] = dict(compare_face(reference, candidate), match_reason=why,
                                   candidate=candidate,
                                   detectability_loss_proxy=(candidate is None) if reference is not None else None)
            faces.append(dict(box=detected['box'], fixed_context_box=[x0,y0,x1-x0,y1-y0],
                              source_landmarks=reference, source_match_reason=reason,
                              source_detection_scale=scale,paired_detection_scale=scale,
                              status='present' if reference is not None else 'unknown',
                              importance=1. if reference is not None else None,
                              metric_scope='face_landmarks_not_identity_includes_detector_noise', states=states))
        return dict(text_digits=texts, face=faces, key_structure='unknown_not_assessed',
                    no_detection_means='unknown_not_absent', labels_transmitted=False)


def reproject_face(row,scale_x,scale_y,detection_scale):
    """Map both boxes and five landmarks from detector pixels to source pixels."""
    import numpy as np
    row=np.asarray(row,dtype=np.float64)
    if row.shape!=(15,) or not np.isfinite(row).all():
        raise ValueError('YuNet must return one finite 15-value face record')
    return dict(box=(row[:4]*[scale_x,scale_y,scale_x,scale_y]).tolist(),
                landmarks=(row[4:14].reshape(5,2)*[scale_x,scale_y]).tolist(),
                confidence=float(row[-1]),detection_scale=float(detection_scale),
                scale_back_xy=[float(scale_x),float(scale_y)])


def assess_clip(source, candidates, provider, sample_id, source_role):
    import numpy as np
    if set(candidates) != set(STATES):
        raise ValueError('exactly B/E/G/EG candidates required; never substitute missing states')
    if source_role not in ('train', 'validation', 'evaluation', 'smoke'):
        raise ValueError('explicit source role required')
    if source.ndim != 4 or source.shape[0] != 17 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError('source must be exactly 17 RGB uint8 frames')
    for state, video in candidates.items():
        if video.shape != source.shape or video.dtype != np.uint8:
            raise ValueError('candidate/source pixel geometry differs: '+state)
    records = []
    for frame in FRAMES:
        began = time.monotonic()
        record = provider.frame(source[frame], {k:v[frame] for k,v in candidates.items()})
        records.append(dict(frame_index=frame, seconds=time.monotonic()-began, **record))
        print(json.dumps(dict(frame=frame, text_regions=len(record['text_digits']),
                              face_regions=len(record['face']), seconds=records[-1]['seconds'])), flush=True)
    return dict(schema=SCHEMA, sample_id=sample_id, source_role=source_role,
                scope='offline_train_evaluation_only', labels_transmitted=False,
                state_order=list(STATES), source_shape=list(source.shape), sampled_frames=list(FRAMES),
                unmeasured_frames='unknown; sparse labels do not certify the whole 17-frame clip',
                importance_definition='unit weight for confidently detected source text/face presence; not task-specific human importance',
                limitations=['source OCR is a pseudo-reference, not ground truth',
                             'OCR edit agreement is not complete semantic preservation',
                             'five facial landmarks measure geometry, not identity',
                             'missed or low-confidence content is unknown, not unimportant',
                             'key subject structure is not assessed',
                             'CRNN_CN has one unlisted output class; any such prediction is unknown'], frames=records)


def region_targets(record, rois):
    """Sparse-frame, positive-only labels; unknown JSON null converts to masked NaN.

    A region is present if an accepted source box intersects it. Candidate error
    is known only if every corresponding accepted box has a known measurement.
    No text/face detections produce unknown, not a negative-importance target.
    """
    importance, status, errors = [], [], []
    observation_loss={category:[] for category in ('text_digits','face')}
    for roi in rois:
        weights, statuses = [], []
        per_state = [[None]*3 for _ in STATES]
        for category_index, (category, field) in enumerate((('text_digits','normalized_edit_error'),
                                                            ('face','normalized_landmark_error'))):
            boxes = [box for frame in record['frames'] for box in frame[category]
                     if box['status'] == 'present' and box_iou(box['box'], roi) > 0]
            weights.append(1. if boxes else None); statuses.append('present' if boxes else 'unknown')
            for state_index, state in enumerate(STATES):
                values = [box['states'][state][field] for box in boxes]
                verified=(category!='text_digits' or all(box.get('source_reference_verified') is True
                          and box.get('candidate_measurements_verified') is True for box in boxes))
                if verified and values and all(value is not None for value in values):
                    per_state[state_index][category_index] = sum(values)/len(values)
            field_proxy='readability_loss_proxy' if category=='text_digits' else 'detectability_loss_proxy'
            per_proxy=[]
            for state in STATES:
                values=[box['states'][state].get(field_proxy) for box in boxes]
                per_proxy.append(sum(float(v) for v in values)/len(values)
                                 if values and all(v is not None for v in values) else None)
            observation_loss[category].append(per_proxy)
        importance.append(weights+[None]); status.append(statuses+['unknown']); errors.append(per_state)
    return dict(category_order=['text_digits','face','key_structure'], state_order=list(STATES),
                importance=importance, annotation_status=status, content_errors=errors,
                rois=rois, temporal_scope='sampled frames 0/8/16 only',
                face_metric_scope='face_landmarks', key_structure_assessed=False,
                separate_observation_loss_proxies=observation_loss,
                observation_proxy_scope='automatic-source OCR confidence/face detectability failure; not semantic-error ground truth',
                text_error_gate='requires BOTH verified source reference and candidate measurement validation; automatic OCR is diagnostic only')


def load_video(spec, default_key):
    import numpy as np
    from PIL import Image
    name, separator, key = str(spec).partition('::')
    path = Path(name)
    if path.is_dir():
        files = sorted(path.glob('*.png'))
        if len(files) != 17:
            raise ValueError('prepared directory must contain exactly 17 PNGs, no hidden temporal selection')
        video = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in files])
        provenance = dict(path=str(path.resolve()), frames={p.name:digest(p) for p in files})
    else:
        with np.load(path, allow_pickle=False) as archive:
            video = archive[key if separator else default_key].copy()
        provenance = dict(path=str(path.resolve()), sha256=digest(path), key=key if separator else default_key)
    provenance['rgb_sha256'] = hashlib.sha256(video.tobytes()).hexdigest()
    provenance['shape'] = list(video.shape)
    return video, provenance


def materialize_paired_region(source, candidates, region):
    """Same region B/E/G/EG, from full B/all_E and saved G/EG cell patches.

    Outside this ONE evaluated region every state remains B. In particular,
    all_E elsewhere is not silently mixed into the isolated E/EG experiment.
    """
    from demo.routervc_visual_router import grid_rois
    if set(candidates) != set(STATES) or candidates['B'].shape != source.shape:
        raise ValueError('paired-region materialization needs full-frame B and all four states')
    rois=grid_rois(*source.shape[1:3])
    if not isinstance(region,int) or not 0<=region<len(rois):
        raise ValueError('paired region must identify one of the existing sixteen cells')
    x,y,w,h=rois[region]
    outputs={'B':candidates['B']}
    for state in ('E','G','EG'):
        video=candidates[state]
        if video.shape == source.shape:
            patch=video[:,y:y+h,x:x+w]
        elif video.shape == (17,h,w,3):
            patch=video
        else:
            raise ValueError('candidate is neither the full frame nor the exact paired-region patch')
        outputs[state]=candidates['B'].copy()
        outputs[state][:,y:y+h,x:x+w]=patch
    return outputs,list(rois[region])


def prepared_sparse_frames(entry):
    """Verify prepared view provenance, reading only the three assessed PNGs."""
    import numpy as np
    from PIL import Image
    view_path=Path(entry['view_json'])
    if digest(view_path)!=entry['view_sha256']:
        raise ValueError('mixed-view metadata changed')
    view=json.loads(view_path.read_text())
    frames_dir=Path(view['frames_dir'])
    paths=sorted(frames_dir.glob('*.png'))
    if not view['complete'] or len(paths)!=17 or str(frames_dir)!=entry['frames_dir']:
        raise ValueError('mixed-view frame directory/count changed')
    if view['view_kind']=='resized_full_frame':
        completion=frames_dir.parent/'complete.json'
        if digest(completion)!=view['fullview_complete_sha256']:
            raise ValueError('full-view preparation record changed')
        record=json.loads(completion.read_text())
        hashes={str((completion.parent/name).resolve()):value for name,value in record['artifacts'].items()}
    elif view['view_kind']=='existing_spatial_crop':
        hashes=view['source_hashes']['files']
    else:
        raise ValueError('unknown source view provenance')
    frames={}; checked={}
    for frame in FRAMES:
        path=paths[frame]; actual=digest(path)
        if hashes.get(str(path.resolve()))!=actual:
            raise ValueError('sampled prepared PNG changed')
        checked[str(path.resolve())]=actual
        with Image.open(path) as image:
            frames[frame]=np.asarray(image.convert('RGB')).copy()
    return view,frames,checked


def coverage_summary(records):
    groups={}
    for record in records:
        key=record['dataset']+'/'+record['view_kind']
        counts=groups.setdefault(key,dict(windows=0,frames=0,text_detection_frames=0,text_presence_proxy_frames=0,
                                         face_presence_proxy_frames=0,half_scale_face_frames=0,
                                         verified_text_reference_frames=0))
        counts['windows']+=1
        for frame in record['frames']:
            counts['frames']+=1
            counts['text_detection_frames']+=bool(frame['text_digits'])
            counts['text_presence_proxy_frames']+=any(t['status']=='present' for t in frame['text_digits'])
            counts['face_presence_proxy_frames']+=any(f['status']=='present' for f in frame['face'])
            counts['half_scale_face_frames']+=any(f['source_detection_scale']==.5 for f in frame['face'])
            counts['verified_text_reference_frames']+=any(t['source_reference_verified'] for t in frame['text_digits'])
    return dict(samples=len(records),groups=groups,
                scope='source-only detector coverage; NO candidates, NO error labels, NO Router training',
                no_detection_means='unknown, not a count of absent content',
                known_failure_cases=['REDS002 Korean plate: confident automatic digits disagree across frames; not verified text',
                                     'Beauty 512 crop fills frame with a face: native-scale YuNet misses it; fixed half-scale can detect'],
                coverage_is_not_precision_or_recall=True)


def scan_coverage(args):
    data=Path(args.data); manifest=data/'complete.json'
    complete=json.loads(manifest.read_text())
    if not complete['complete']:
        raise ValueError('mixed-view preparation is incomplete')
    entries=complete['samples'][:args.limit] if args.limit else complete['samples']
    if not entries:
        raise ValueError('no prepared samples')
    binding=dict(data=str(manifest.resolve()),data_sha256=digest(manifest),
                 samples=entries,code_sha256=digest(__file__),assets_sha256=digest(args.assets/'assets.json'),
                 scope='source-only coverage, not training',sampled_frames=list(FRAMES))
    args.output.mkdir(parents=True,exist_ok=True)
    request=args.output/'request.json'
    if request.exists() and json.loads(request.read_text())!=binding:
        raise ValueError('coverage bound inputs changed; use a new output')
    if not request.exists(): save(request,binding)
    began=time.monotonic(); provider=None; records=[]; artifacts={}
    for entry in entries:
        sid=entry['sample_id']
        if not isinstance(sid,str) or Path(sid).name!=sid:
            raise ValueError('invalid sample id')
        view,frames,checked=prepared_sparse_frames(entry)
        sample_binding=dict(view_json=entry['view_json'],view_sha256=entry['view_sha256'],frame_hashes=checked,
                            request_sha256=digest(request))
        output=args.output/'samples'/sid/'coverage.json'
        marker=output.with_name('complete.json')
        if marker.exists():
            completed=json.loads(marker.read_text())
            if completed['binding']!=sample_binding or completed['coverage_sha256']!=digest(output):
                raise ValueError('completed coverage sample changed')
            record=json.loads(output.read_text())
        else:
            if time.monotonic()-began>args.max_hours*3600:
                save(args.output/'progress.json',dict(complete=False,completed=len(records),total=len(entries),
                                                     deadline_reached=True,**resource_record()))
                return
            if provider is None: provider=OpenCVProvider(args.assets)
            started=time.monotonic()
            observed=[dict(frame_index=i,**provider.frame(image,{})) for i,image in frames.items()]
            record=dict(sample_id=sid,dataset=view['sample']['dataset'],view_kind=view['view_kind'],
                        router_split=view['router_split'],frames=observed,scope='source_only_coverage',
                        no_candidate_inference=True,labels_transmitted=False,seconds=time.monotonic()-started)
            save(output,record)
            save(marker,dict(complete=True,binding=sample_binding,coverage_sha256=digest(output)))
        records.append(record)
        artifacts[str(output.relative_to(args.output))]=digest(output)
        save(args.output/'progress.json',dict(complete=False,completed=len(records),total=len(entries),
                                             seconds_this_attempt=time.monotonic()-began,**resource_record()))
        print(json.dumps(dict(source_coverage_completed=len(records),total=len(entries),sample=sid)),flush=True)
    summary=coverage_summary(records)
    summary.update(binding=binding,artifacts=artifacts)
    target=args.output/'summary.json'
    if target.exists() and json.loads(target.read_text())!=summary:
        raise ValueError('completed coverage summary changed')
    if not target.exists(): save(target,summary)
    done=args.output/'complete.json'
    if not done.exists():
        save(done,dict(complete=True,summary_sha256=digest(target),seconds_this_attempt=time.monotonic()-began,
                       **resource_record()))
    elif json.loads(done.read_text())['summary_sha256']!=digest(target):
        raise ValueError('coverage completion changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch-assets'); fetch.add_argument('--assets', type=Path, default=ASSETS)
    coverage=sub.add_parser('coverage',help='source-only coverage, no four-state generation or training')
    coverage.add_argument('--data',type=Path,required=True,help='prepared mixedview root with complete.json')
    coverage.add_argument('--assets',type=Path,default=ASSETS)
    coverage.add_argument('--output',type=Path,required=True)
    coverage.add_argument('--limit',type=int,default=0)
    coverage.add_argument('--max-hours',type=float,default=2.)
    label = sub.add_parser('label')
    label.add_argument('--assets', type=Path, default=ASSETS)
    label.add_argument('--source', required=True)
    label.add_argument('--candidate', action='append', required=True, help='B|E|G|EG=NPZ[::key] or prepared PNG directory')
    label.add_argument('--sample-id', required=True)
    label.add_argument('--source-role', choices=('train','validation','evaluation','smoke'), required=True)
    label.add_argument('--output', type=Path, required=True)
    label.add_argument('--paired-region',type=int,help='same-region B/all_E/G-patch/EG-patch smoke; only this ROI is assessed')
    args = parser.parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('downloads and offline scanning must run under tmux')
    if args.command == 'fetch-assets':
        with output_lock(args.assets): fetch_assets(args.assets)
        return
    if args.command=='coverage':
        if args.limit<0 or not math.isfinite(args.max_hours) or args.max_hours<=0:
            raise ValueError('limit must be nonnegative and max-hours finite positive')
        with output_lock(args.output): scan_coverage(args)
        return
    with output_lock(args.output):
        label_command(args)


def label_command(args):
    specifications = {}
    for argument in args.candidate:
        state, separator, path = argument.partition('=')
        if not separator or state not in STATES or state in specifications:
            raise ValueError('exactly one path per B/E/G/EG state required')
        specifications[state] = path
    if set(specifications) != set(STATES):
        raise ValueError('exactly B/E/G/EG candidates required')
    source, source_provenance = load_video(args.source, 'source')
    candidates, provenance = {}, {}
    for state, spec in specifications.items():
        candidates[state], provenance[state] = load_video(spec, 'reconstruction')
    paired_roi=None
    if args.paired_region is not None:
        candidates,paired_roi=materialize_paired_region(source,candidates,args.paired_region)
    binding = dict(source=source_provenance, candidates=provenance,
                   sample_id=args.sample_id, source_role=args.source_role,
                   code_sha256=digest(__file__), assets_sha256=digest(args.assets/'assets.json'),
                   paired_region=args.paired_region,paired_roi=paired_roi,
                   candidate_construction='isolated same region; outside region all states B' if paired_roi else 'supplied full-frame states')
    args.output.mkdir(parents=True, exist_ok=True)
    done = args.output/'complete.json'
    if done.exists():
        existing = json.loads(done.read_text())
        if existing['binding'] != binding or existing['labels_sha256'] != digest(args.output/'labels.json'):
            raise ValueError('completed labels changed; use a new output')
        print(json.dumps(dict(verified=True, resumed_without_inference=True))); return
    began = time.monotonic()
    provider = OpenCVProvider(args.assets)
    labels = assess_clip(source, candidates, provider, args.sample_id, args.source_role)
    from demo.routervc_visual_router import grid_rois
    rois = [paired_roi] if paired_roi else grid_rois(*source.shape[1:3])
    labels.update(binding=binding, region_targets=region_targets(labels,rois),
                  opencv_version=provider.cv.__version__, provider_device='CPU',
                  text_confidence_threshold=provider.text_threshold, face_detection_threshold=provider.face_threshold)
    save(args.output/'labels.json',labels)
    save(done,dict(complete=True,binding=binding,labels_sha256=digest(args.output/'labels.json'),
                   seconds=time.monotonic()-began,**resource_record()))
    print(json.dumps(dict(complete=True,output=str(args.output),seconds=time.monotonic()-began)),flush=True)


if __name__ == '__main__':
    main()
