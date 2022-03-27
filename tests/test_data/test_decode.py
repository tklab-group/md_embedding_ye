import sys
sys.path.append('../../')
from common.util import decode_module_data, get_file_level_info

# module_data = 'src/tklab/hagward/lcextractor/scm/MappedList#public_void_setReplaced(T_oldItem,T_newItem)'
module_data = 'org.apache.catalina.tribes.transport.nio.PooledParallelSenderMBean#public_int_getRxBufSize()'
print(decode_module_data(module_data))
print(get_file_level_info(module_data))
