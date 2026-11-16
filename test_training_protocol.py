import unittest
from training_protocol import development_records, require_development_video


class TrainingBoundaryTests(unittest.TestCase):
    def test_development_contains_only_original_train_videos(self):
        fit, validation = development_records()
        self.assertEqual((len(fit), len(validation)), (320, 80))
        self.assertTrue(all(r['split'] == 'train' for r in fit + validation))
        self.assertFalse({r['video_id'] for r in fit} & {r['video_id'] for r in validation})

    def test_official_test_id_is_rejected_for_fit_and_validation(self):
        for role in ('fit', 'validation'):
            with self.assertRaises(ValueError):
                require_development_video('vid0012.mp4', role)

    def test_validation_id_cannot_be_exported_as_training(self):
        _, validation = development_records()
        video_id = validation[0]['video_id']
        require_development_video(video_id, 'validation')
        with self.assertRaises(ValueError):
            require_development_video(video_id, 'fit')


if __name__ == '__main__':
    unittest.main()
