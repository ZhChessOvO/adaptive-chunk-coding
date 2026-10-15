import copy
import unittest

from tools.g_objective_trial import recipe
from tools.g_objective_trial_evaluate import indexed, aggregate, ARMS, NAMES


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(loss=dict(rgb_temporal=.1,lpips=.5,latent_temporal=.2),phase_weight=1.,
                        rows=[dict(dataset='REDS'),dict(dataset='UVG'),dict(dataset='REDS')],
                        initial=dict(path='untouched'),code=dict(old='immutable'))

    def test_only_coefficient_and_variant_differ(self):
        before=copy.deepcopy(self.old)
        a,b=(recipe(self.old,n) for n in ARMS)
        self.assertEqual(a.pop('variant'),'control');self.assertEqual(b.pop('variant'),'relaxed_rgb_temporal')
        self.assertEqual(a['loss'].pop('rgb_temporal'),.1);self.assertEqual(b['loss'].pop('rgb_temporal'),.025)
        self.assertEqual(a,b);self.assertEqual(self.old,before)
        self.assertEqual(a['phase_weight'],1.);self.assertEqual(a['arms'],['mixed'])
        self.assertEqual(a['steps'],400);self.assertEqual(a['learning_rate']['warmup'],40)

    def test_smoke_is_paired_and_includes_both_datasets(self):
        for arm in ARMS:
            config=recipe(self.old,arm,True)
            self.assertEqual(config['steps'],6)
            self.assertEqual([r['dataset'] for r in config['rows']],['REDS','UVG'])
            self.assertEqual(config['seed'],2026101503)

    def test_unknown_recipe_rejected(self):
        with self.assertRaises(ValueError):recipe(self.old,'other')

    def item(self):
        rows=[]
        for n in NAMES:
            for size in (256,512):
                q=dict(lpips_alex=.2,psnr_db=25.,temporal_delta_mae=.01)
                rows.append(dict(name=n,size=size,crop=[0,0,size,size],quality=q,wide_384_quality=q,
                    phase_error=dict(phase_rms_u8=.5),observation=dict(noise='same',condition='same')))
        return dict(case=dict(evaluation_group='internal_validation',dataset='REDS'),results=rows)

    def test_pair_validation_and_grouping(self):
        item=self.item();self.assertEqual(len(indexed(item)),6)
        groups=aggregate([item,item]);self.assertEqual(len(groups),6)
        self.assertTrue(all(r['count']==2 for r in groups))
        self.assertTrue(all(r['lpips']==.2 for r in groups))
        self.assertTrue(all(('wide_384_quality' in r)==(r['size']==512) for r in groups))

    def test_missing_duplicate_or_changed_conditions_rejected(self):
        for change in ('missing','duplicate','noise','condition','crop'):
            item=self.item()
            if change=='missing':item['results'].pop()
            elif change=='duplicate':item['results'].append(copy.deepcopy(item['results'][-1]))
            elif change=='crop':item['results'][-1]['crop'][0]=8
            else:item['results'][-1]['observation'][change]='different'
            with self.assertRaises(ValueError):aggregate([item])


if __name__=='__main__':unittest.main()
