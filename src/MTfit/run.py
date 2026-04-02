"""
run.py
******
Core module for MTfit - handles all the command line parsing logic and calling the forward model based inversion.
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

from pathlib import Path
import subprocess
import warnings
import traceback

try:
    import pyqsub
    _PYQSUB = True
except Exception:
    _PYQSUB = False

from .inversion import Inversion
from .inversion import combine_mpi_output
from .utilities.extensions import get_extensions
from .extensions import default_pre_inversions
from .extensions import default_post_inversions
from .extensions import default_extensions
from .utilities.argparser import MTfit_parser


WARNINGS_MAP = {'a': 'always', 'd': 'default',
                'e': 'error', 'i': 'ignore',
                'm': 'module', 'o': 'once'}


ERROR_MESSAGE = """
**********************************************
Error running MTfit:

If this is a recurring error, please post an issue at https://github.com/djpugh/MTfit/issues/ (if one doesn't already exist)
including the traceback and the following information to help with diagnosis:

## Environment Information

```
{}
```

## Traceback

```
{}
```

**********************************************
"""


def MTfit(data: dict | None = None, data_file: str | list | bool = False,
          location_pdf_file_path: str | list | bool = False,
          algorithm: str = 'Time', parallel: bool = True, n: int = 0,
          phy_mem: int = 8, dc: bool = False, **kwargs) -> int:
    """
    Runs MTfit

    Creates an MTfit.inversion.Inversion object for the given arguments and runs the forward model based inversion.
    """
    if data is None:
        data = {}
    try:
        kwargs['data'] = data
        kwargs['data_file'] = data_file
        kwargs['location_pdf_file_path'] = location_pdf_file_path
        kwargs['algorithm'] = algorithm
        kwargs['parallel'] = parallel
        kwargs['n'] = n
        kwargs['phy_mem'] = phy_mem
        kwargs['dc'] = dc
        # GET PLUGINS
        pre_inversion_names, pre_inversions = get_extensions('MTfit.pre_inversion', default_pre_inversions)
        post_inversion_names, post_inversions = get_extensions('MTfit.post_inversion', default_post_inversions)
        extension_names, extensions = get_extensions('MTfit.extension', default_extensions)
        for ext in extensions.values():
            result = ext(**kwargs)
            if result != 1:
                return result
        if kwargs.get('combine_mpi_output', False):
            combine_mpi_output(kwargs.get('path', ''), kwargs.get('output_format', 'matlab'), **kwargs)
            return 0
        if len(data) or (isinstance(data_file, str) and not Path(data_file).is_dir()) or isinstance(data_file, list):
            warnings.filterwarnings(WARNINGS_MAP[kwargs.get('warnings', 'd')])
            for plugin in pre_inversions.values():
                kwargs = plugin(**kwargs)
            if not kwargs.get('_mpi_call', False):
                print('Running MTfit.')
            inversion = Inversion(**kwargs)
            inversion.forward()
            if kwargs.get('dc_mt', False):
                kwargs['dc'] = not kwargs['dc']
                inversion = Inversion(**kwargs)
                inversion.forward()
            for plugin in post_inversions.values():
                plugin(**kwargs)
            return 0
    except Exception as e:
        from . import get_details_json
        print(ERROR_MESSAGE.format(get_details_json(), traceback.format_exc()))
        raise e


def run(args: list[str] | None = None) -> int:
    """
    Runs inversion from command line arguments.
    """
    options, options_map = MTfit_parser(args)
    if options['qsub'] and _PYQSUB:
        options_map['data_file'] = options_map['DATAFILE']
        options['singlethread'] = not options.pop('parallel')
        options['_mpi_call'] = options['mpi']
        if 'mcmc' in options['algorithm'].lower():
            options['max_samples'] = options['chain_length']
        return pyqsub.submit(options, options_map, __name__)
    else:
        for key in list(options.keys()):
            if 'qsub' in key:
                options.pop(key)
        if options['mpi'] and not options['_mpi_call']:
            try:
                import mpi4py  # noqa: F401
            except Exception:
                raise ImportError('MPI module mpi4py not found, unable to run in mpi')
            options['_mpi_call'] = True
            print('Running MTfit using mpirun')
            optstring = pyqsub.make_optstr(options, options_map)
            mpiargs = ["mpirun", "-n", str(options['n']), "MTfit"]
            mpiargs.extend(optstring.split())
            return subprocess.call(mpiargs)
        return MTfit(**options)
