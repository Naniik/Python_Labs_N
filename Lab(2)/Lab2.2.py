import hashlib
def generate_file_hashes(*file_paths):
    result_hash = {}
    for path in file_paths:
        try:
         with open(path,'rb') as file:
             readed_file = file.read()
             print (readed_file)
             hasher = hashlib.sha256(readed_file).hexdigest()
             print(hasher)
             result_hash[path]=hasher
        except FileNotFoundError:
            print(f"Помилка: файл не знайдено.")
        except IOError:
            print(f"Помилка при читанні файлу")
    return result_hash
result = generate_file_hashes("file1_forlb2.txt","file2_forlb2.txt")
print(result)