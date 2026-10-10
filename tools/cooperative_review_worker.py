"""Run the pinned P2 decoder with an additional independent sender-access guard."""
import os
from pathlib import Path
import sys


def sender_guard(event, values):
    if event == 'open' and isinstance(values[0], (str, bytes)):
        path = str(Path(os.fsdecode(values[0])).resolve())
        if '/routervc_latent_sender_' in path or '/routervc_sender_' in path:
            raise RuntimeError('review receiver attempted sender artifact access')


if __name__ == '__main__':
    sys.addaudithook(sender_guard)
    from tools.cooperative_worker import main
    main()
