#This program emulates the functionality of TensorFlow:
#It never works, and instead outputs a random error.

#!/bin/python3

from random import randint
import time

time.sleep(4)

Err = [
r"""Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorflow'""",

r"""Traceback (most recent call last):
File "C:\Users\User\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 74, in
raise ImportError(msg)
File "C:\Users\User\AppData\Local\Programs\Python\Python35\lib\importlib_init_.py", line 126, in import_module
return _bootstrap._gcd_import(name[level:], package, level)
File "", line 986, in _gcd_import
File "", line 969, in _find_and_load
ImportError: DLL load failed with error code -1073741795

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
File "C:\Users\User\AppData\Local\Programs\Python\Python35\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in
from tensorflow.python.pywrap_tensorflow_internal import *
File "C:\Users\User\AppData\Local\Programs\Python\Python35\lib\importlib_init.py", line 126, in import_module
return _bootstrap._gcd_import(name[level:], package, level)
ImportError: No module named 'TensorFlow'

Failed to load the native TensorFlow runtime.""",

r"""2019-05-25 21:22:20.872655: E tensorflow/core/common_runtime/process_function_library_runtime.cc:764] Component function execution failed: Unknown: Fail to find the dnn implementation.
[[node bidirectional_2/CudnnRNN]]
[[loss_2/dense_5_loss/binary_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/else/_1/has_valid_nonscalar_shape/then/_47/has_invalid_dims_0/_52]]
2019-05-25 21:22:20.875125: E tensorflow/stream_executor/cuda/cuda_dnn.cc:338] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
2019-05-25 21:22:20.875145: W tensorflow/core/framework/op_kernel.cc:1431] OP_REQUIRES failed at cudnn_rnn_ops.cc:1280 : Unknown: Fail to find the dnn implementation.
2019-05-25 21:22:20.875197: E tensorflow/core/common_runtime/process_function_library_runtime.cc:764] Component function execution failed: Unknown: Fail to find the dnn implementation.
[[node bidirectional_2/CudnnRNN]]

""" + '\x1b[1;31;40m' + """Please upgrade to the latest version by using pip install --upgrade cudnn
If you are currently using the latest version of CUDNN, please downgrade to an earlier version.""" + '\x1b[0m',

r"""Traceback (most recent call last):
File "C:\Users\USER\Anaconda3\lib\site-packages\tensorflow_core\python\pywrap_tensorflow.py", line 58, in
from tensorflow.python.pywrap_tensorflow_internal import *
File "C:\Users\USER\Anaconda3\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 28, in
_pywrap_tensorflow_internal = swig_import_helper()
File "C:\Users\USER\Anaconda3\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
_mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
File "C:\Users\USER\Anaconda3\lib\imp.py", line 242, in load_module
return load_dynamic(name, filename, file)
File "C:\Users\USER\Anaconda3\lib\imp.py", line 342, in load_dynamic
return _load(spec)
ImportError: DLL load failed: The specified module could not be found.

""" + '\x1b[1;31;40m' + 'Tensorflow2 is incompatible with this version of Python, please upgrade to the latest version.  If you are currently using the latest version of Python, please downgrade to an earlier version.' + '\x1b[0m',

r"""Traceback (most recent call last):
File "C:\Users\USER\Anaconda3\lib\site-packages\tensorflow_core\python\pywrap_tensorflow.py", line 58, in
from tensorflow.python.pywrap_tensorflow_internal import *
File "C:\Users\USER\Anaconda3\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 28, in
_pywrap_tensorflow_internal = swig_import_helper()
File "C:\Users\USER\Anaconda3\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
_mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
File "C:\Users\USER\Anaconda3\lib\imp.py", line 242, in load_module
return load_dynamic(name, filename, file)
File "C:\Users\USER\Anaconda3\lib\imp.py", line 342, in load_dynamic
return _load(spec)
ImportError: DLL load failed: The specified module could not be found.

""" + '\x1b[1;31;40m' + 'Tensorflow is incompatible with this version of Tensorflow, please upgrade to the latest version.  If you are currently using the latest version of Tensorflow, please downgrade to an earlier version.' + '\x1b[0m',

r"""2019-05-25 21:22:20.872655: E tensorflow/core/common_runtime/process_function_library_runtime.cc:764] Component function execution failed: Unknown: Fail to find the dnn implementation.
[[node bidirectional_2/CudnnRNN]]
[[loss_2/dense_5_loss/binary_crossentropy/weighted_loss/broadcast_weights/assert_broadcastable/is_valid_shape/else/_1/has_valid_nonscalar_shape/then/_47/has_invalid_dims_0/_52]]
2019-05-25 21:22:20.875125: E tensorflow/stream_executor/cuda/cuda_dnn.cc:338] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
2019-05-25 21:22:20.875145: W tensorflow/core/framework/op_kernel.cc:1431] OP_REQUIRES failed at cudnn_rnn_ops.cc:1280 : Unknown: Fail to find the dnn implementation.
2019-05-25 21:22:20.875197: E tensorflow/core/common_runtime/process_function_library_runtime.cc:764] Component function execution failed: Unknown: Fail to find the dnn implementation.
[[node bidirectional_2/CudnnRNN]]

""" + '\x1b[1;31;40m' + """Please upgrade to the latest version by using pip install --upgrade CUDA
If you are currently using the latest version of CUDA, please downgrade to an earlier version.""" + '\x1b[0m',

r"""Traceback (most recent call last):
File "C:\Users\USER\Anaconda3\lib\site-packages\tensorflow_core\python\pywrap_tensorflow.py", line 58, in
from tensorflow.python.pywrap_tensorflow_internal import *
File "C:\Users\USER\Anaconda3\lib\imp.py", line 342, in load_dynamic
return _load(spec)
Error: No Module Found: Tensorflow.Usability

""" + '\x1b[1;31;40m' + 'Tensorflow is incompatible with this computer.  Please purchase a newer, better computer.  If this computer is already top of the line, please purchase an older computer because Tensorflow has not caught up with this technology yet.' + '\x1b[0m',

]



max = len(Err) - 1

chosen = randint(0,max)
print(Err[chosen])

