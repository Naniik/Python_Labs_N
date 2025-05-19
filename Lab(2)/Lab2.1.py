import re
def analyze_log_file(log_file_path):
    response={}
    try:
        with open(log_file_path,'r') as file:
            for line in file:
                match = re.search(r'"[^"]*" (\d{3})', line)
                if match:
                    code = match.group(1)
                    response[code] = response.get(code, 0) + 1
    except FileNotFoundError:
        print(f"Помилка: файл '{log_file_path}' не знайдено.")
    except IOError:
        print(f"Помилка при читанні файлу '{log_file_path}'.")
    return response
result=analyze_log_file(r"D:\Хлам_Стундента\Python_Labs_N\apache_logs.txt")
print(result)