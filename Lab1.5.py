'''
.encode() перетворює рядок на байти
.hexdigest() повертає хеш у шістнадцятковому форматі
.digest() повертає байтове представлення хешу
'''
import hashlib
users = {
    'naniik':{
        'password': hashlib.md5('Nastya12345'.encode()).hexdigest(),
        'full_name': 'Ткаченко Анастасія Максимівна'
    },
    'echo_raven42':{
        'password':hashlib.md5('Myterious#Raven'.encode()).hexdigest(),
        'full_name':'Шевченко Павло Юрійович'
    },
    'mary_smith': {
        'password': hashlib.md5('securepass456'.encode()).hexdigest(),
        'full_name': 'Смітко Марія Степанівна'
    },
    'ninideutsch':{
        'password': hashlib.md5('13145116JKLi'.encode()).hexdigest(),
        'full_name': 'Монастирська Ніна Віталіївна'
    }
}
def add_user():
    login = input("Введіть логін нового користувача:")
    if login in users:
        print("Користувач з таким логіном вже існує")
        return
    full_name = input("Введіть повне ім'я:")
    password = input("Введіть пароль:")
# Хешування паролю за допомогою MD5
    password_hash = hashlib.md5(password.encode()).hexdigest()
#print(f"Хеш вашого паролю: {password_hash}")
    users[login] = {
    'password': password_hash,
    'full_name':full_name
}
print("Користувач був успішно доданий!")

def authentificate_user():
    login = input ("Введіть логін:")
    if login not in users:
        print("Користувача з таким логіном не знайдено")
        return
    password = input ("Введіть пароль:")
    password_hash = hashlib.md5(password.encode()).hexdigest()
#print(f"Хеш введеного паролю: {password_hash}")
    if password_hash == users[login]['password']:
        print(f"Вітаємо, {users[login]['full_name']}!")
    else:
        print("Невірний пароль")

def show_users():
    print("\nСписок зареєстрованих користувачів:")
    for login, user_info in users.items():
        print(f"Логін:{login},Повне ПІБ: {user_info['full_name']}")

def main():
    while True:
        print("\n --- Система Аутентифікації ---")
        print("1. Авторизація")
        print("2. Реєстрація нового користувача")
        print("3. Показати список користувачів")
        print("4. Вихід")
        choice = input ("Виберіть дію (1-4):")
        if choice =='1':
            authentificate_user()
        elif choice == '2':
            add_user()
        elif choice == '3':
            show_users()
        elif choice == '4':
            print("До побачення!")
            break
        else:
            print("Невірний вибір.Спробуйте ще раз")
main()