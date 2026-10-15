import unittest
import torch
from routervc.generation.objective_diagnostic import gradient_statistics
from tools.g_objective_report import aggregate


class GradientTests(unittest.TestCase):
    def test_conflict_is_signed(self):
        result=gradient_statistics(dict(lpips=torch.tensor([1.,0.]),reconstruction=torch.tensor([-2.,0.])))
        self.assertEqual(result['cosine'][0][1],-1.)
        self.assertEqual(result['norms']['reconstruction'],2.)
        self.assertEqual(result['total_projection']['lpips'],-1.)

    def test_orthogonal_and_projection_sum(self):
        result=gradient_statistics(dict(a=torch.tensor([1.,0.]),b=torch.tensor([0.,2.])))
        self.assertEqual(result['cosine'][0][1],0.)
        self.assertAlmostEqual(sum(result['total_projection'].values()),1.)

    def test_finite_positive_required(self):
        for x in (torch.zeros(2),torch.tensor([float('nan'),1.])):
            with self.assertRaises(ValueError):gradient_statistics(dict(a=x))

    def test_exact_cancellation_not_percentages(self):
        result=gradient_statistics(dict(a=torch.ones(2),b=-torch.ones(2)))
        self.assertEqual(result['total_norm'],0.)
        self.assertIsNone(result['total_projection']['a'])

    def test_report_preserves_conflicting_examples(self):
        cases=[]
        for sign in (-1.,1.):
            stats=gradient_statistics(dict(lpips=torch.tensor([1.,0.]),edge=torch.tensor([sign,0.])))
            cases.append(dict(gradient=stats,values=dict(lpips=dict(weighted=.2),edge=dict(weighted=.01))))
        row=aggregate(cases)[1]
        self.assertEqual(row['negative_lpips_cosine'],1)
        self.assertEqual(row['mean_lpips_cosine'],0.)
        self.assertEqual(row['median_norm_over_lpips'],1.)

    def test_empty_report_rejected(self):
        with self.assertRaises(ValueError):aggregate([])


if __name__=='__main__':unittest.main()
