import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def method_ast(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.dump(item, include_attributes=False)
    raise AssertionError('%s.%s not found in %s' % (class_name, method_name, path))


class MoseiP4PortTest(unittest.TestCase):
    def test_fixed_budget_objective_matches_mosi(self):
        mosi = method_ast(
            ROOT / 'MOSI/models/model.py', 'TVA_fusion', 'get_budgeted_auxiliary'
        ).replace('MOSI', 'DATASET')
        mosei = method_ast(
            ROOT / 'MOSEI/models/model.py', 'TVA_fusion', 'get_budgeted_auxiliary'
        ).replace('MOSEI', 'DATASET')
        self.assertEqual(mosei, mosi)

    def test_mosei_model_contains_interventional_path(self):
        source = (ROOT / 'MOSEI/models/model.py').read_text()
        required = [
            'TemporalReliabilityHead',
            'ordinal_reliability_pair',
            'scheduled_corruption_progress',
            'reliability_allocation_control',
            'self.budget_warmup_mode',
            "reliability_v['q_task']",
            "reliability_a['q_task']",
        ]
        for token in required:
            self.assertIn(token, source)

    def test_mosei_training_updates_and_logs_reliability(self):
        source = (ROOT / 'MOSEI/train/TVA_train.py').read_text()
        self.assertIn('model.vision_reliability.parameters()', source)
        self.assertIn('model.audio_reliability.parameters()', source)
        self.assertIn('Interventional reliability epoch %d:', source)

    def test_cli_allows_mosei_but_not_unported_sims(self):
        source = (ROOT / 'run_experiment.py').read_text()
        self.assertIn("args.dataset not in {'MOSI', 'MOSEI'}", source)
        self.assertIn('currently supports MOSI and MOSEI', source)


if __name__ == '__main__':
    unittest.main()
