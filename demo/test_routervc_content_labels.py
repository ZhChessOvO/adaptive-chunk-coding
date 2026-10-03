"""Offline provider contract tests; no models, network, GPU, or dataset required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from demo import routervc_content_labels as labels


class ContentLabelsTest(unittest.TestCase):
    def test_subprocess_mirror_cleanup_does_not_modify_parent_or_proxy(self):
        original = {'HF_ENDPOINT':'mirror','PIP_INDEX_URL':'mirror','http_proxy':'transport',
                    'SOME_MIRROR':'mirror','HF_HOME':'cache','CUDA_VISIBLE_DEVICES':'0'}
        cleaned,removed = labels.clean_download_environment(original)
        self.assertEqual(set(removed),{'HF_ENDPOINT','PIP_INDEX_URL','SOME_MIRROR'})
        self.assertEqual(cleaned['http_proxy'],'transport')
        self.assertEqual(cleaned['HF_HOME'],'cache')
        self.assertIn('HF_ENDPOINT',original)

    def test_ctc_blank_separates_repeat_and_literal_dash_survives(self):
        # blank, A,A,blank,A,- -> AA-; '-' is a real charset symbol, not blank.
        logits = np.full((6,1,3),-20.)
        for row,index in enumerate((0,1,1,0,1,2)):
            logits[row,0,index] = 20
        result = labels.ctc_decode(logits,'A-')
        self.assertEqual(result['text'],'AA-')
        self.assertGreater(result['confidence'],.999)
        self.assertEqual(labels.ctc_decode(np.asarray([[.99,.005,.005]]),'A-')['text'],'')

    def test_ctc_checks_charset_and_finite_values(self):
        with self.assertRaises(ValueError):
            labels.ctc_decode(np.zeros((2,5)),'A-')
        with self.assertRaises(ValueError):
            labels.ctc_decode(np.asarray([[0,np.nan,0]]),'A-')

    def test_unlisted_upstream_output_class_is_unknown_not_fabricated(self):
        result=labels.ctc_decode(np.asarray([[-20.,-20.,-20.,20.]]),'A-')
        self.assertEqual(result['confidence'],0.)
        self.assertEqual(result['text'],'')
        self.assertEqual(result['reason'],'unmapped_upstream_output_class')

    def test_known_ocr_digit_change_separate_from_readability_loss(self):
        source = dict(text='公交A18',confidence=.99)
        known = labels.compare_text(source,dict(text='公交A19',confidence=.99))
        self.assertEqual(known['normalized_edit_error'],.2)
        self.assertTrue(known['digits_changed'])
        self.assertFalse(known['readability_loss_proxy'])
        missing = labels.compare_text(source,dict(text='',confidence=0.))
        self.assertEqual(missing['status'],'unknown')
        self.assertIsNone(missing['normalized_edit_error'])
        self.assertIsNone(missing['digits_changed'])
        self.assertTrue(missing['readability_loss_proxy'])

    def test_unreliable_source_is_not_ground_truth(self):
        result = labels.compare_text(dict(text='8',confidence=.4),dict(text='9',confidence=.99))
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['readability_loss_proxy'])

    def test_face_measures_five_landmarks_not_identity(self):
        reference = dict(box=[0,0,30,40],landmarks=[[5,5],[20,5],[12,10],[7,20],[18,20]])
        result = labels.compare_face(reference,reference)
        self.assertEqual(result['normalized_landmark_error'],0.)
        self.assertFalse(result['face_identity_assessed'])
        candidate = copy.deepcopy(reference)
        candidate['landmarks'] = (np.asarray(candidate['landmarks'])+[3,4]).tolist()
        self.assertAlmostEqual(labels.compare_face(reference,candidate)['normalized_landmark_error'],.1)
        self.assertIsNone(labels.compare_face(reference,None)['normalized_landmark_error'])

    def test_ambiguous_face_match_unknown(self):
        faces = [dict(box=[0,0,30,40]),dict(box=[1,0,30,40])]
        self.assertIsNone(labels.match_face(faces,[0,0,30,40])[0])
        self.assertIsNone(labels.match_face([], [0,0,30,40])[0])

    def test_region_no_detection_unknown_not_absent(self):
        result = labels.region_targets(dict(frames=[dict(text_digits=[],face=[])]),[[0,0,100,100]])
        self.assertEqual(result['annotation_status'],[['unknown']*3])
        self.assertEqual(result['importance'],[[None]*3])
        self.assertEqual(result['content_errors'],[[[None]*3 for _ in labels.STATES]])

    def test_region_partial_measurement_not_cherry_picked_and_other_types_unknown(self):
        text = dict(status='present',box=[10,10,20,10],
                    source_reference_verified=True,candidate_measurements_verified=True,
                    states={state:dict(normalized_edit_error=.2) for state in labels.STATES})
        other = copy.deepcopy(text); other['states']['G']['normalized_edit_error'] = None
        record = dict(frames=[dict(text_digits=[text,other],face=[])])
        result = labels.region_targets(record,[[0,0,100,100],[100,0,100,100]])
        self.assertEqual(result['annotation_status'][0],['present','unknown','unknown'])
        self.assertEqual(result['importance'][0],[1.,None,None])
        self.assertIsNone(result['content_errors'][0][2][0])
        self.assertEqual(result['content_errors'][0][0][0],.2)
        self.assertEqual(result['annotation_status'][1],['unknown']*3)

    def test_missing_readable_candidate_keeps_separate_observation_risk(self):
        text=dict(status='present',box=[10,10,20,10],states={
            state:dict(normalized_edit_error=None,readability_loss_proxy=(state in ('G','EG')))
            for state in labels.STATES})
        result=labels.region_targets(dict(frames=[dict(text_digits=[text],face=[])]),[[0,0,64,64]])
        self.assertIsNone(result['content_errors'][0][2][0])
        self.assertEqual(result['separate_observation_loss_proxies']['text_digits'],[[0.,0.,1.,1.]])
        self.assertEqual(result['separate_observation_loss_proxies']['face'],[[None]*4])

    def test_automatic_ocr_agreement_never_becomes_verified_character_error(self):
        text=dict(status='present',box=[0,0,20,20],source_reference_verified=False,
                  candidate_measurements_verified=False,
                  states={s:dict(normalized_edit_error=.5) for s in labels.STATES})
        record=dict(frames=[dict(text_digits=[text],face=[])])
        result=labels.region_targets(record,[[0,0,64,64]])
        self.assertEqual(result['importance'][0][0],1.)  # Presence proxy only.
        self.assertIsNone(result['content_errors'][0][0][0])
        text['source_reference_verified']=True
        self.assertIsNone(labels.region_targets(record,[[0,0,64,64]])['content_errors'][0][0][0])

    def test_face_box_and_landmarks_reproject_to_original_coordinates(self):
        row=np.asarray([10,20,30,40, 12,22, 18,22, 15,26, 13,30, 17,30, .95])
        result=labels.reproject_face(row,2.,3.,.5)
        self.assertEqual(result['box'],[20.,60.,60.,120.])
        self.assertEqual(result['landmarks'][0],[24.,66.])
        self.assertEqual(result['detection_scale'],.5)

    def test_half_scale_fallback_only_if_native_missing_and_forced_scale_locked(self):
        provider=labels.OpenCVProvider.__new__(labels.OpenCVProvider)
        frame=np.zeros((64,64,3),np.uint8)
        with patch.object(provider,'_detect_faces_at_scale',side_effect=[[{'face':1}],[]]) as detect:
            self.assertEqual(provider.detect_faces(frame),[{'face':1}])
            self.assertEqual(detect.call_count,1)
        with patch.object(provider,'_detect_faces_at_scale',side_effect=[[],[{'face':2}]]) as detect:
            self.assertEqual(provider.detect_faces(frame),[{'face':2}])
            self.assertEqual([call.args[1] for call in detect.call_args_list],[1.,.5])
        with patch.object(provider,'_detect_faces_at_scale',return_value=[]) as detect:
            provider.detect_faces(frame,forced_scale=.5)
            self.assertEqual(detect.call_count,1)
            self.assertEqual(detect.call_args.args[1],.5)

    def test_output_lock_rejects_concurrent_writer_then_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with labels.output_lock(directory):
                with self.assertRaises(RuntimeError):
                    with labels.output_lock(directory): pass
            with labels.output_lock(directory): pass

    def test_coverage_counts_are_detection_coverage_not_absence_or_accuracy(self):
        record=dict(dataset='UVG',view_kind='existing_spatial_crop',frames=[
            dict(text_digits=[],face=[dict(status='present',source_detection_scale=.5)]),
            dict(text_digits=[],face=[])])
        result=labels.coverage_summary([record])
        self.assertTrue(result['coverage_is_not_precision_or_recall'])
        self.assertEqual(result['groups']['UVG/existing_spatial_crop']['face_presence_proxy_frames'],1)
        self.assertEqual(result['groups']['UVG/existing_spatial_crop']['verified_text_reference_frames'],0)

    def test_source_coverage_resumes_without_inference_and_rejects_changed_record(self):
        class Provider:
            calls=0
            def __init__(self,*a): pass
            def frame(self,source,candidates):
                self.calls+=1
                self.assert_no_candidates=not candidates
                return dict(text_digits=[],face=[])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=root/'data'; data.mkdir(); assets=root/'assets'; assets.mkdir()
            labels.save(assets/'assets.json',{})
            labels.save(data/'complete.json',dict(complete=True,samples=[
                dict(sample_id='one',view_json='/one',view_sha256='one',frames_dir='/one/frames')]))
            args=SimpleNamespace(data=data,assets=assets,output=root/'out',limit=0,max_hours=1.)
            view=dict(sample={'dataset':'REDS'},view_kind='resized_full_frame',router_split='train')
            sparse={i:np.zeros((8,8,3),np.uint8) for i in labels.FRAMES}
            provider=Provider()
            with patch.object(labels,'prepared_sparse_frames',return_value=(view,sparse,{'frame':'hash'})), \
                    patch.object(labels,'OpenCVProvider',return_value=provider):
                labels.scan_coverage(args)
                before=(args.output/'samples/one/coverage.json').read_bytes()
                labels.scan_coverage(args)
                self.assertEqual(provider.calls,3)
                self.assertTrue(provider.assert_no_candidates)
                self.assertEqual((args.output/'samples/one/coverage.json').read_bytes(),before)
                labels.save(args.output/'samples/one/coverage.json',dict(changed=True))
                with self.assertRaisesRegex(ValueError,'sample changed'):
                    labels.scan_coverage(args)

    def test_raw_upstream_charset_backslash_is_not_python_line_escape(self):
        characters='0123456789Aa中\\'+''.join(chr(i) for i in range(0x5000,0x5000+3930))
        self.assertEqual(len(characters),3944)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'crnn.py'
            path.write_text("CHARSET_CN_3944 = '''\n"+'\n'.join(characters)+"'''\n")
            self.assertEqual(labels.charset_from_official(path),characters)

    def test_paired_region_uses_codec_grid_and_base_outside_every_state(self):
        # 72 is divisible by 8 but not 32: naive equal quarters would misalign.
        source=np.full((17,72,80,3),9,np.uint8)
        base=np.ones_like(source)
        enhanced=np.full_like(source,2)
        patch=np.full((17,16,24,3),3,np.uint8)
        # Column 1 spans x16..40 under the exact aligned codec grid.
        output,roi=labels.materialize_paired_region(source,dict(B=base,E=enhanced,G=patch,EG=patch+1),1)
        self.assertEqual(roi,[16,0,24,16])
        for key in ('E','G','EG'):
            self.assertTrue(np.all(output[key][:,16:]==1))
        self.assertTrue(np.all(output['E'][:,:16,16:40]==2))
        self.assertTrue(np.all(output['G'][:,:16,16:40]==3))
        self.assertTrue(np.all(output['EG'][:,:16,16:40]==4))
        self.assertTrue(np.all(enhanced==2))

    def test_fixed_sparse_frames_and_no_missing_state_substitution(self):
        class Provider:
            def __init__(self): self.seen=[]
            def frame(self,source,candidates):
                self.seen.append(int(source[0,0,0]))
                return dict(text_digits=[],face=[])
        source = np.repeat(np.arange(17,dtype=np.uint8)[:,None,None,None],3,axis=3)
        provider = Provider()
        result = labels.assess_clip(source,{s:source for s in labels.STATES},provider,'synthetic','smoke')
        self.assertEqual(provider.seen,[0,8,16])
        self.assertEqual(result['sampled_frames'],[0,8,16])
        self.assertFalse(result['labels_transmitted'])
        with self.assertRaises(ValueError):
            labels.assess_clip(source,{'B':source},provider,'synthetic','smoke')
        with self.assertRaises(ValueError):
            labels.assess_clip(source,{s:source[:1] for s in labels.STATES},provider,'synthetic','smoke')


if __name__ == '__main__':
    unittest.main()
