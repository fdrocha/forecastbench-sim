import time
import os
from freeciv_world.configs import fc_args

if fc_args['debug.logging_path'] is not None:
    logging_dir = os.path.expanduser(fc_args['debug.logging_path'])
else:
    # Default recording/logging root: worlds/freeciv/logs.
    #
    # This MUST resolve to the same root that worlds/freeciv/scripts/*
    # (run_world.py savegame downloads, serializers) use. A post-restructure
    # split — observations under worlds/logs while savegames/serializers used
    # worlds/freeciv/logs — made serialization silently savegame-blind and
    # published dark worlds. run_world.py imports `logging_dir` from here, so
    # there is a single source of truth; pin debug.logging_path to relocate
    # BOTH observation states and savegames together.
    logging_dir = os.path.normpath(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..', '..', 'logs'))

# Level for the freeciv_world file logger. DEBUG logs every packet, which has
# filled pods (37 GB/pod). Set FBSIM_FC_LOG_LEVEL=WARNING (or INFO) to tame it.
_FC_FILE_LOG_LEVEL = os.environ.get('FBSIM_FC_LOG_LEVEL', 'DEBUG')

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname) -3s [%(filename)s:%(lineno)d] %(message)s'
        },
    },
    'handlers': {
        'rootFileHandler': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(logging_dir, f"{time.strftime('%Y.%m.%d')}_root.log"),
            'mode': 'w',
        },
        'civrealmFileHandler': {
            'level': _FC_FILE_LOG_LEVEL,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            # Change time.strftime('%%Y.%%m.%%d') to time.strftime('%%Y.%%m.%%d_%%H:%%M:%%S') to create a new file for each script
            'filename': os.path.join(logging_dir, f"{time.strftime('%Y.%m.%d')}_{fc_args['username']}.log"),
            'mode': 'a',
        },
        'consoleHandler': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'handlers': ['rootFileHandler'],
            'level': 'INFO',
            'propagate': False
        },
        'root': {
            'handlers': ['rootFileHandler'],
            'level': 'INFO',
            'propagate': False
        },
        'freeciv_world': {
            'handlers': ['civrealmFileHandler'],
            'level': _FC_FILE_LOG_LEVEL,
            'propagate': False
        },
    }
}
