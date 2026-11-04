import copy
import unittest

from scripts.report_full_adaptation import require_replay_coverage


class AdaptationReportGatesTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(status='ok', record_id='fit1', prediction_sha256='p1',
                          transform_sha256='t1', transform_file='transform1')]
        self.classifiers = {'fit1': dict(prediction_sha256='p1', transform_sha256='t1', max_prediction_error=0.)}
        self.transforms = {'transform1': dict(transform_sha256='t1', max_array_error=0., stop_reason='objective_tolerance')}

    def test_rejects_missing_and_stale_replays(self):
        for classifiers in [{}, {'fit1': dict(self.classifiers['fit1'], prediction_sha256='old')}]:
            with self.assertRaises((AssertionError, KeyError)):
                require_replay_coverage(self.rows, classifiers, self.transforms)
        with self.assertRaises(KeyError):
            require_replay_coverage(self.rows, self.classifiers, {})

    def test_rejects_unconverged_or_inaccurate_transform(self):
        for changed in [dict(stop_reason='max_iter'), dict(max_array_error=1e-3), dict(max_array_error=float('nan'))]:
            transforms = copy.deepcopy(self.transforms)
            transforms['transform1'].update(changed)
            with self.assertRaises(AssertionError):
                require_replay_coverage(self.rows, self.classifiers, transforms)

    def test_failed_validation_is_retained_but_incomplete_fit_rejected(self):
        rows = self.rows + [dict(status='failed')]
        self.assertEqual(require_replay_coverage(rows, self.classifiers, self.transforms), dict(classifiers=1, transforms=1))
        rows[0] = dict(rows[0], status='incomplete')
        with self.assertRaises(AssertionError):
            require_replay_coverage(rows, self.classifiers, self.transforms)
