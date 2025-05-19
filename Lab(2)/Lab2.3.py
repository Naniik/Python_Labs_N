def filter_ips(input_file_path, output_file_path, allowed_ips):
    ip_addresses = {}
    try:
        with open(input_file_path, 'r') as file:
            for line in file:
                ip = line.split()[0]
                if ip in allowed_ips:
                    if ip in ip_addresses:
                        ip_addresses[ip] += 1
                    else:
                        ip_addresses[ip] = 1
    except FileNotFoundError:
        print(f"Помилка: файл '{input_file_path}' не знайдено.")
        return
    except IOError:
        print(f"Помилка при читанні файлу '{input_file_path}'.")
        return
    try:
        with open(output_file_path, 'w') as file:
            for ip, count in ip_addresses.items():
                file.write(f"{ip} - {count}\n")
    except IOError:
        print(f"Помилка при записі у файл '{output_file_path}'.")
allowed_ips = ["208.115.111.72", "144.76.194.187"]
filter_ips(
    input_file_path=r"D:\Хлам_Стундента\Python_Labs_N\apache_logs.txt",
    output_file_path="filtered_output.txt",
    allowed_ips=allowed_ips
)



