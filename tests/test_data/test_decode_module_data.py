import sys
sys.path.append('../../')
from common.util import decode_module_data, delete_method_modifiers, decode_method_signature, split_param, leave_one_out

# module_data = 'src/tklab/hagward/lcextractor/scm/MappedList#void_setReplaced(T_oldItem,T_newItem)'
module_data = 'org.apache.accumulo.server.rpc.RpcWrapperTest#private_fake_proc[FakeService]_createProcessFunction(String_methodName,boolean_isOneway...)'
# module_data = 'jakarta.el.Util#private_[T]_Wrapper[T]_findWrapper(Class[#]_clazz,List[Wrapper[T]]_wrappers,String_name,Class[#][]_paramTypes,Object[]_paramValues)'
package, class_name, method_signature = decode_module_data(module_data)
print(delete_method_modifiers(method_signature))
print(decode_method_signature(method_signature))
# test_param1 = 'Map[String,T]_fragmentResources,Map[String,T]_mainResources,Map[String,T]_tempResources,Map[String,Boolean]_mergeInjectionFlags,WebXml_fragment'
# test_param2 = 'Map[String,List[String]]_respHeaders'
# print('split_param', split_param(test_param1))
# print('split_param', split_param(test_param2))

tran = [1, 2, 3, 4]
print(leave_one_out(tran))
