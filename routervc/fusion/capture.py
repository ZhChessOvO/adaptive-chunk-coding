"""Observe existing G's already computed halo, without changing its inference."""
from unittest.mock import patch
import numpy as np

from demo.four_state_receive import PersistentRGB
from demo import scalable_cooperation_format as fmt


class CaptureRGB(PersistentRGB):
    def __call__(self, pixels, settings):
        if len(settings['generate']) != 1 or settings['window'] != 17:
            raise ValueError('capture supports one current 17-frame G call')
        captured = []
        original = self.model.restore

        def observe(*args, **kwargs):
            frames, runtime = original(*args, **kwargs)
            captured.append(np.stack(frames))
            return frames, runtime

        with patch.object(self.model, 'restore', observe):
            current, runtime = super().__call__(pixels, settings)
        if len(captured) != 1 or len(runtime['windows']) != 1:
            raise ValueError('unexpected temporal tiling')
        raw = captured[0]
        x0, y0, cw, ch = runtime['windows'][0]['crop']
        t, n, x, y, w, h = settings['generate'][0]
        if raw.shape != (n, ch, cw, 3):
            raise ValueError('raw halo shape differs')
        restored = pixels.copy()
        restored[t:t+n, y:y+h, x:x+w] = raw[:, y-y0:y-y0+h, x-x0:x-x0+w]
        replay = fmt.combine(pixels, restored, fmt.weights(pixels.shape, settings))
        np.testing.assert_array_equal(replay, current)
        return current, raw, dict(runtime=runtime, crop=[x0, y0, cw, ch],
                                  core=[x, y, w, h], feather_replay_exact=True)
