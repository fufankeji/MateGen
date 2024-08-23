
import sys
from io import StringIO

def execute_python_code(code: str) -> str:
    try:
        # 重定向标准输出
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        exec(code)  # 执行代码，不限制内建函数
        sys.stdout = old_stdout
        return redirected_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return str(e)