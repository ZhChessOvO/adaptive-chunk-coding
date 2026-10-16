import copy
import unittest

from tools.g_bounded_finish_report import paired, grouped


def metric(lpips):
    return dict(quality=dict(lpips_alex=lpips, psnr_db=25., temporal_delta_mae=3., rgb_mse=100.),
                G_core_phase=dict(per_frame_rms_mean=1.))


def summary():
    reviews = []
    for step in (400, 800):
        rows = []
        for sid, cohort, dataset in (('a', 'diagnostic', 'REDS'), ('b', 'diagnostic', 'UVG'),
                                     ('c', 'grouped_validation', 'UVG')):
            row = dict(sample_id=sid, cohort=cohort, dataset=dataset, step=step,
                actual_bytes=100, bpp=.1, regions=[1], resources={},
                references={k: metric(.2) for k in ('received', 'legacy', 'control')}, **metric(.18))
            if step == 800:
                row['quality']['lpips_alex'] = .17
                row['G_core_phase']['per_frame_rms_mean'] = 1.2
            rows.append(row)
        reviews.append(dict(step=step, cases=rows))
    return dict(reviews=reviews)


class FinishReportTests(unittest.TestCase):
    def test_paired_by_id_and_separate_internal_cohort(self):
        data = summary(); data['reviews'][1]['cases'].reverse()
        groups = grouped(paired(data))
        self.assertEqual(len(groups), 3)
        for row in groups:
            self.assertEqual(row['count'], 1)
            self.assertEqual(row['lpips_improved'], 1)
            self.assertEqual(row['phase_increased'], 1)
            self.assertAlmostEqual(row['models']['step800']['lpips_alex'], .17)

    def test_changed_bytes_or_references_rejected(self):
        for key, value in (('actual_bytes', 101), ('regions', [2]), ('references', {})):
            data = summary(); data['reviews'][1]['cases'][0][key] = value
            with self.assertRaises(ValueError):
                paired(data)

    def test_missing_or_duplicate_case_rejected(self):
        data = summary(); data['reviews'][1]['cases'].pop()
        with self.assertRaises(ValueError):
            paired(data)
        data = summary(); data['reviews'][1]['cases'].append(copy.deepcopy(data['reviews'][1]['cases'][0]))
        with self.assertRaises(ValueError):
            paired(data)

    def test_wrong_milestone_rejected(self):
        data = summary(); data['reviews'][1]['step'] = 600
        with self.assertRaises(ValueError):
            paired(data)


if __name__ == '__main__':
    unittest.main()
