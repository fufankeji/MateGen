
import sys
from io import StringIO

def execute_python_code(code: str) -> str:
    try:
        # �ض����׼���
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        exec(code)  # ִ�д��룬�������ڽ�����
        sys.stdout = old_stdout
        return redirected_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return str(e)