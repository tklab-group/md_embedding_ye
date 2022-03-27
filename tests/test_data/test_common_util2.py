import sys

sys.path.append('../../')
from common.util import decode_module_data, delete_method_modifiers, decode_method_signature, split_param, camel_case_split

module_data_list = [
    'org.apache.tomcat.util.threads.DedicatedThreadExecutor#public_[V]_V_execute(Callable[V]_callable)',
    'org.apache.catalina.core.StandardContext#public_ServletRegistration.Dynamic_dynamicServletAdded(Wrapper_wrapper)',
    'jakarta.el.Util#private_[T]_Wrapper[T]_findWrapper(Class[#]_clazz,List[Wrapper[T]]_wrappers,String_name,Class[#][]_paramTypes,Object[]_paramValues)',
    'org.apache.catalina.session.TestPersistentManagerIntegration#public_void_noSessionCreate_57637()'
]
for i in range(len(module_data_list)):
    module_data = module_data_list[i]
    print('module data', module_data)
    package, class_name, method_signature = decode_module_data(module_data)
    print(package, class_name, method_signature)
    class_name_result = camel_case_split(class_name)
    print(class_name, class_name_result)
    # print(delete_method_modifiers(method_signature))
    return_type, method_name, split_param_list = decode_method_signature(
        method_signature=method_signature,
        is_delete_modifier=True,
        # return type
        is_delete_void_return_type=True,
        is_number_type_token_return_type=True,
        # method name
        is_delete_number_method_name=True,
        # param type
        is_number_type_token_param_type=True,
        # param name
        is_delete_number_param_name=True,
        is_delete_single_token_param_name=True,
    )
    print(return_type, method_name, split_param_list)
    print()

