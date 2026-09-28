import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from demo import scalable_cooperation_format as c
from demo import scalable_generation_format as old
from demo.scalable_format import parse
from demo import test_scalable_generation as legacy_tests


class CooperationTests(unittest.TestCase):
    def fixture(self):
        inner,control = legacy_tests.GenerationTests().fixture()
        return inner,dict(control,processing_scale=1,blend=1.)

    def test_roundtrip_and_every_complete_prefix(self):
        inner,control = self.fixture()
        wire = c.wrap(inner,control)
        loaded,received,parsed,offset = c.parse(wire)
        self.assertEqual(control,loaded)
        self.assertEqual(inner,received)
        for boundary in [parsed.base_end]+[p.end_offset for p in parsed.packets]:
            self.assertEqual(c.wrap(inner[:boundary],control),wire[:offset+boundary])
        with self.assertRaises(ValueError):
            c.parse(wire[:-1])
        self.assertFalse(c.parse(wire[:-1],allow_incomplete_tail=True)[2].packets)

    def test_versions_do_not_silently_change_meaning(self):
        inner,control = self.fixture()
        legacy = {k:v for k,v in control.items() if k in old.KEYS}
        v1,v2 = old.wrap(inner,legacy),c.wrap(inner,control)
        with self.assertRaises(ValueError):
            c.parse(v1)
        with self.assertRaises(ValueError):
            old.parse(v2)
        self.assertEqual(old.parse(v1)[0],legacy)
        before = old.weights((33,128,128,3),legacy,parse(inner))
        after = c.weights((33,128,128,3),control)
        self.assertEqual(before[2,80,80],0)
        self.assertEqual(after[2,80,80],1)

    def test_four_modes_and_protection(self):
        inner,control = self.fixture()
        base = np.full((33,128,128,3),20,np.uint8)
        enhanced = base.copy()
        enhanced[1:9,64:,64:] = 40
        generated = np.full_like(base,80)
        alpha = c.weights(base.shape,control)
        result = c.combine(enhanced,generated,alpha)
        self.assertTrue(np.all(result[:,:32,:32] == base[:,:32,:32]))
        self.assertTrue(np.all(c.combine(base,generated,np.zeros_like(alpha)) == base))
        self.assertTrue(np.all(c.combine(enhanced,generated,np.zeros_like(alpha)) == enhanced))
        self.assertEqual(result[2,80,80,0],80)
        self.assertEqual(result[20,80,80,0],80)
        control["blend"] = .5
        result = c.combine(enhanced,generated,c.weights(base.shape,control))
        self.assertEqual(result[2,80,80,0],60)  # no double addition of E

    def test_empty_generation_and_zero_blend(self):
        inner,control = self.fixture()
        control["generate"] = []
        self.assertEqual(c.parse(c.wrap(inner,control))[0],control)
        self.assertFalse(np.any(c.weights((33,128,128,3),control)))
        control["generate"] = [[0,33,0,0,128,128]]
        control["blend"] = 0
        self.assertFalse(np.any(c.weights((33,128,128,3),control)))

    def test_invalid_controls_and_corruption(self):
        inner,control = self.fixture()
        for key,value in (("blend",float("nan")),("blend",1.1),("processing_scale",3),
                          ("processing_scale",True),("strength",-.1),("hidden",1),
                          ("generate",[[0,16,0,0,128,128]])):
            with self.assertRaises(ValueError):
                c.wrap(inner,dict(control,**{key:value}))
        wire = c.wrap(inner,control)
        for index in (0,4,15,len(wire)-1):
            bad = bytearray(wire)
            bad[index] ^= 1
            with self.assertRaises(ValueError):
                c.parse(bytes(bad))
        control["generate"] = []
        control["protect"] = [[0,33,120,120,32,32]]
        with self.assertRaises(ValueError):
            c.wrap(inner,control)

    def test_prefix_changes_condition_but_not_blending_support(self):
        inner,control = self.fixture()
        parsed = parse(inner)
        first = c.parse(c.wrap(inner[:parsed.base_end],control))
        last = c.parse(c.wrap(inner,control))
        self.assertEqual(first[0],last[0])
        self.assertEqual(first[2].base,last[2].base)
        np.testing.assert_array_equal(c.weights((33,128,128,3),first[0]),
                                      c.weights((33,128,128,3),last[0]))

    def test_disabled_generation_never_resolves_generator_assets(self):
        # Receiver fallback must work even when the optional G weights are absent.
        from demo import scalable_cooperation_decode as decoder
        from demo.scalable_format import frame_hash
        inner,control = self.fixture()
        parsed = parse(inner)
        base = np.full((33,128,128,3),20,np.uint8)
        enhanced = base.copy()
        enhanced[1:9,64:,64:] = 40
        report = dict(source_frames_read=False,applied_packets=[1],
            base_bytes=len(parsed.base),container_header_bytes=parsed.base_end-len(parsed.base),
            packet_bytes=sum(len(p.wire) for p in parsed.packets),incomplete_tail_bytes=0,
            total_bytes=len(inner),base_hash=frame_hash(base),output_hash=frame_hash(enhanced),
            non_enhanced_exact=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for disabled,blend in ((True,1.),(False,0.)):
                stream = root/f"{disabled}-{blend}.acsg"
                stream.write_bytes(c.wrap(inner,dict(control,blend=blend)))
                args = SimpleNamespace(stream=stream,output=root/stream.stem,
                                       disable_generation=disabled)
                with patch.multiple(decoder,configure_torch=lambda:None,
                    load_model=lambda _:object(),BaseCodec=lambda *_:object(),
                    decode_enhancement=lambda *a,**kw:(enhanced,dict(report),base)), \
                    patch.object(decoder,"identities",side_effect=FileNotFoundError("G weights absent")) as assets, \
                    patch.object(decoder,"restore",side_effect=AssertionError("G executed")) as restore, \
                    patch.object(decoder.torch.cuda,"max_memory_allocated",return_value=0), \
                    patch.object(decoder.torch.cuda,"empty_cache"), \
                    patch.object(decoder.torch.distributed,"is_initialized",return_value=False):
                    decoder.decode(args)
                assets.assert_not_called()
                restore.assert_not_called()
                result = json.loads((args.output/"decode.json").read_text())
                self.assertFalse(result["generation_executed"])
                self.assertFalse(result["generation_assets_validated"])
                with np.load(args.output/"reconstruction.npz") as value:
                    np.testing.assert_array_equal(value["reconstruction"],enhanced)


if __name__ == "__main__":
    unittest.main()
