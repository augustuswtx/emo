import json
import tempfile
import unittest
from pathlib import Path

from build_actionability_evidence_manifest import build_manifest, sha256


class ActionabilityEvidenceManifestTest(unittest.TestCase):
    def test_missing_sources_remain_explicit_and_are_not_imputed(self):
        result = build_manifest({})
        self.assertEqual(result['evidence'], [])
        self.assertEqual(result['sources']['c2_primary']['status'], 'missing')
        self.assertEqual(result['coverage_by_level']['allocation_actionability'], [])

    def test_p6_p7_provenance_and_levels_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = {
                'seed': 1111,
                'checkpoint_sha256': 'a' * 64,
            }
            metrics = {
                'n': 100, 'auroc': 0.51, 'auprc': 0.55,
                'spearman_score_utility': 0.01,
                'top_quarter_harmful_rate': 0.44,
                'positive_prevalence': 0.52,
            }
            p6 = root / 'p6.json'
            p6.write_text(json.dumps({
                'design': 'linear feasibility baseline for sparse visual utility labels',
                'split': 'complete batch-pair holdout',
                'test_sources': [source],
                'models': {'all': metrics},
            }), encoding='utf-8')
            p7 = root / 'p7.json'
            p7.write_text(json.dumps({
                'design': 'repeated-meta stability and fixed predictor repair screen',
                'sources': [source],
                'stability': {
                    'samples': 768,
                    'first_context_matches_mean_rate': 0.641,
                    'mean_context_sign_agreement': 0.496,
                    'fraction_majority_at_least_five_sixths': 0.215,
                    'context_reliability_ratio': 0.171,
                },
                'models': {'mlp_all': metrics},
            }), encoding='utf-8')
            result = build_manifest({'p6': p6, 'p7': p7})
            self.assertEqual(result['sources']['p6']['sha256'], sha256(p6))
            self.assertEqual(result['sources']['p7']['sha256'], sha256(p7))
            self.assertEqual(result['coverage_by_level']['update_stability'], ['p7'])
            self.assertEqual(result['coverage_by_level']['utility_predictability'],
                             ['p6', 'p7'])
            stability = [item for item in result['evidence']
                         if item['level'] == 'update_stability']
            self.assertEqual(len(stability), 4)
            self.assertTrue(all(item['checkpoint_sha256'] == 'a' * 64
                                for item in stability))

    def test_raw_label_link_checks_summary_seed_checkpoint_and_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = root / 'p7-records.json'
            records.write_text(json.dumps({
                'seed': 1111, 'checkpoint_sha256': 'a' * 64,
                'records': [{'samples': [{}, {}]}],
            }), encoding='utf-8')
            summary = root / 'p7-summary.json'
            summary_data = {
                'design': 'repeated-meta stability and fixed predictor repair screen',
                'sources': [{
                    'file': records.name, 'seed': 1111,
                    'checkpoint_sha256': 'a' * 64,
                }],
                'stability': {
                    'samples': 2,
                    'first_context_matches_mean_rate': 0.5,
                    'mean_context_sign_agreement': 0.5,
                    'fraction_majority_at_least_five_sixths': 0.5,
                    'context_reliability_ratio': 0.2,
                },
                'models': {},
            }
            summary.write_text(json.dumps(summary_data), encoding='utf-8')
            result = build_manifest({'p7': summary, 'p7_records': records})
            self.assertEqual(
                result['sources']['p7_records']['summary_link_check']['raw_sample_count'],
                2,
            )
            summary_data['stability']['samples'] = 3
            summary.write_text(json.dumps(summary_data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'sample count'):
                build_manifest({'p7': summary, 'p7_records': records})

    def test_aggregate_uncertainty_and_condition_are_structured(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'f4.csv'
            path.write_text(
                'corruption,severity,modality,auroc,auroc_sd,delta_mae,delta_mae_sd\n'
                'Temporal shift,1.00,Vision,0.51,0.01,-0.002,0.0005\n',
                encoding='utf-8',
            )
            result = build_manifest({'f4': path})
            auroc, delta = result['evidence']
            self.assertEqual(auroc['condition'],
                             {'corruption': 'Temporal shift', 'severity': 1.0})
            self.assertEqual(auroc['sd'], 0.01)
            self.assertEqual(delta['sd'], 0.0005)
            self.assertEqual(delta['value'], -0.002)


if __name__ == '__main__':
    unittest.main()
