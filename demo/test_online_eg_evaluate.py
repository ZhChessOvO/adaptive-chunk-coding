"""CPU regressions for real-byte accounting, pairing, prefixes and no-recompute resume."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from demo import compact_enhancement_format as compact
from demo import online_eg_evaluate as evaluate
from demo.online_eg_eval_core import split_prefix, prefix_pixels, noise_pair, CLIPS
from demo.online_eg_report import common_rate
from demo.scalable_codec import atomic_json, atomic_npz, file_hash
from demo.scalable_format import parse, frame_hash


def container():
    meta = dict(compact.CONSTANTS, width=32, height=32, frame_count=17,
                enhancement_codec='uf_feature_head_pad16_v2', **{k:'0'*64 for k in compact.HASHES})
    wire = compact.base_container(b'base',meta)
    for i,(t,n) in enumerate(((0,1),(1,8),(9,8)),1):
        wire += compact.packet_bytes(dict(packet_id=i,start=t,count=n,roi=[0,0,16,16],qstep=1.,
            codec=meta['enhancement_codec']),b'fakepayload',codec=meta['enhancement_codec'])
    return wire


def report_noise():
    return dict(generation_runtime=dict(windows=[dict(region=0,start=0,crop=[0,0,256,256])],
        condition_windows=[dict(before_vae='a',after_vae='b',before_diffusion='c',
            diffusion_noise=dict(dtype='torch.bfloat16',sha256='noise'),conditions=['rgb'])]))


class PlanTests(unittest.TestCase):
    def test_count_unique_and_complete(self):
        rows=[dict(sample_id=sid,name=j['name']) for i,sid in enumerate(CLIPS) for j in evaluate.jobs(i==0)]
        self.assertEqual(len(evaluate.grouped(rows)),92)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):evaluate.grouped(bad)

    def test_crossed_models_are_explicit(self):
        plan={j['name']:j for j in evaluate.jobs()}
        self.assertEqual((plan['cross_oldE_newG']['e_arm'],plan['cross_oldE_newG']['g_arm']),('fixed','joint'))
        self.assertEqual((plan['cross_newE_fixedG']['e_arm'],plan['cross_newE_fixedG']['g_arm']),('joint','fixed'))

    def test_every_case_has_both_direct_and_generated_outputs(self):
        for case in evaluate.CASES:
            selected=[j for j in evaluate.jobs() if j['case']==case and not j['name'].startswith('cross')]
            self.assertEqual({(j['kind'],j['e_arm']) for j in selected},{(k,a) for k in ('E','G') for a in evaluate.ARMS})

    def test_different_condition_allows_paired_noise(self):
        a,b=report_noise(),report_noise();b['generation_runtime']['condition_windows'][0]['conditions']=['new RGB']
        noise_pair(a,b)
        with self.assertRaises(AssertionError):noise_pair(a,b,same_condition=True)

    def test_noise_or_window_changes_rejected(self):
        for what in ('noise','window'):
            a,b=report_noise(),report_noise()
            if what=='noise':b['generation_runtime']['condition_windows'][0]['diffusion_noise']['sha256']='other'
            else:b['generation_runtime']['windows'][0]['start']=8
            with self.assertRaises(AssertionError):noise_pair(a,b)

    def test_identical_conditions_optional_check(self):
        noise_pair(report_noise(),report_noise(),same_condition=True)


class PrefixTests(unittest.TestCase):
    def test_literal_prefix_all_boundaries(self):
        wire=container();last=b''
        for n in range(4):
            value=split_prefix(wire,n)
            self.assertTrue(wire.startswith(value));self.assertTrue(value.startswith(last))
            self.assertEqual(len(parse(value).packets),n);last=value
        self.assertEqual(last,wire)

    def test_prefix_bounds(self):
        for n in (-1,4):
            with self.assertRaises(ValueError):split_prefix(container(),n)

    def test_only_received_regions_are_modified(self):
        base=np.zeros((17,32,32,3),np.uint8);full=np.full_like(base,255)
        result=prefix_pixels(base,full,split_prefix(container(),2))
        expected=base.copy();expected[:9,:16,:16]=255
        np.testing.assert_array_equal(result,expected)
        np.testing.assert_array_equal(base,np.zeros_like(base))


class ResumeTests(unittest.TestCase):
    def test_completed_point_skips_decoder_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sid='sample'
            job=dict(name='fixed_E_none',case='none',e_arm='fixed',g_arm=None,kind='E')
            prepared=root/'evaluation'/sid/'prepared/fixed';prepared.mkdir(parents=True)
            wire=split_prefix(container(),0)
            pixels=np.zeros((17,32,32,3),np.uint8)
            (prepared/'none.acse').write_bytes(wire)
            atomic_npz(prepared/'none_expected.npz',reconstruction=pixels)
            atomic_json(prepared/'complete.json',dict(base_hash='base',artifacts={},
                records=dict(none=dict(output_hash=frame_hash(pixels)))))
            folder=root/'evaluation'/sid/job['name'];folder.mkdir()
            (folder/'stream.acse').write_bytes(wire)
            atomic_npz(folder/'reconstruction.npz',reconstruction=pixels)
            d=dict(source_frames_read=False,total_bytes=len(wire),base_hash='base',non_enhanced_exact=True,
                output_hash=frame_hash(pixels),base_bytes=4,container_header_bytes=len(wire)-4,
                packet_bytes=0,incomplete_tail_bytes=0)
            atomic_json(folder/'decode.json',d)
            record=dict(job=job,bytes=len(wire),stream_file='stream.acse',fresh_decode=d,
                artifacts={n:file_hash(folder/n) for n in ('stream.acse','decode.json','reconstruction.npz')})
            atomic_json(folder/'result.json',record)
            before=(folder/'result.json').read_bytes()
            def forbidden():raise AssertionError('metric recomputed')
            with patch.object(evaluate,'execute',side_effect=AssertionError('decoded again')):
                actual=evaluate.point(root,dict(sample=dict(sample_id=sid)),job,{}, {},None,forbidden)
            self.assertEqual(actual,record);self.assertEqual((folder/'result.json').read_bytes(),before)
            (folder/'stream.acse').write_bytes(wire+b'changed')
            with self.assertRaises(RuntimeError):
                evaluate.point(root,dict(sample=dict(sample_id=sid)),job,{}, {},None,forbidden)

    def test_same_q_is_not_assumed_same_rate(self):
        rows=[dict(sample_id=sid,name=j['name'],bytes=100+(i*5)) for i,sid in enumerate(CLIPS) for j in evaluate.jobs(i==0)]
        rows[-1]['bytes']+=100
        self.assertEqual(len(evaluate.grouped(rows)),92)


class RateTests(unittest.TestCase):
    def curve(self,rates,offset=0):
        return [dict(bytes=b,roi_quality=dict(lpips_alex=.4-.02*np.log(b)+offset,
                    psnr_db=20+.1*np.log(b),temporal_delta_mae=1.)) for b in rates]

    def test_common_rate_only_within_overlap(self):
        value=common_rate(self.curve([100,200,400]),self.curve([200,400,800],-.01),'roi_quality')
        self.assertEqual(value['overlap_bytes'],[200,400])
        self.assertAlmostEqual(value['bytes'],np.sqrt(200*400))
        self.assertAlmostEqual(value['lpips_gain'],.01)

    def test_no_overlap_no_extrapolation(self):
        self.assertIsNone(common_rate(self.curve([100,200]),self.curve([300,400]),'roi_quality'))

    def test_duplicate_rates_do_not_select_best_quality(self):
        self.assertIsNone(common_rate(self.curve([100,100,400]),self.curve([200,300,400]),'roi_quality'))


if __name__=='__main__':unittest.main()
