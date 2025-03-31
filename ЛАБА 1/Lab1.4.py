'''
.update() - це вбудований метод у Python для словників, який дозволяє додавати або оновлювати елементи в словнику. Це простий спосіб перетворити список словників в один словник.
Альтернативний варіант без .update() виглядав би так:
pythonCopydef manage_task_status(tasks):
    task_statuses = {}
    for task_dict in tasks:
        for key, value in task_dict.items():
            task_statuses[key] = value
    return task_statuses
'''
task_list = [
    {"Зварити борщ":"виконано"},
    {"Зробити лабораторну з Програмування":"в процесі"},
    {"Прочитати Холлі С.Кінг":"очікує"},
    {"Піти по воду":"очікує"},
    {"Нагодувати кота":"виконано"}
]
def manage_task_status(tasks):
    task_statuses = {}
    for task in tasks:
        task_statuses.update(task)
    return task_statuses

#Виклик функції для отримання статусів задач
result_statuses = manage_task_status(task_list)
print ("\nПоточні статуси задач:", result_statuses)

def update_task_status(tasks, task_name, new_status):
    if task_name in tasks:
        tasks[task_name] = new_status
        print("Статус задачі {} змінено на {}".format(task_name, new_status))
    else:
        print("Задача {} не знайдена".format(task_name))
    return tasks
def add_task(tasks, task_name, status ="очікує"):
    if task_name not in tasks:
        tasks[task_name] = status
        print("Задача {} додана зі статусом {}".format(task_name, status))
    else:
        print("Задача {} вже існує".format(task_name))
    return tasks

def remove_task(tasks, task_name):
    if task_name in tasks:
        del tasks[task_name]
        print("Задача {} видалена".format(task_name))
    else:
        print("Задача {} не знайдена".format(task_name))
    return tasks

print("Поточні задачі:")
for task, status in result_statuses.items():
    print("{}:{}".format(task,status))

tasks = add_task(result_statuses, "Купити книгу")
tasks = update_task_status(tasks, "Купити книгу", "виконано")
tasks = remove_task(tasks, "Прочитати Холлі С.Кінг")

waiting_tasks = []
for task, status in result_statuses.items():
    if status == "очікує":
        waiting_tasks.append(task)
print("\nСписок задач із статусом 'очікує':", waiting_tasks)
print("\nОновлені задачі:")
for task,status in tasks.items():
    print("{}:{}".format(task, status))
#print(f"{task}:{status}")