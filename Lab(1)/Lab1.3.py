
sells_inventory = [
    {"продукт": "ноутбук", "кількість": 4, "ціна": 1000},
    {"продукт": "смартфон", "кількість": 2, "ціна": 500},
    {"продукт": "планшет", "кількість": 3, "ціна": 400},
    {"продукт": "монітор", "кількість": 5, "ціна": 50}
]

def calculate_income (sells):
    goods_income={}
    for sell in sells_inventory:
        good = sell["продукт"]
        quantity = sell["кількість"]
        price = sell["ціна"]
        if good in goods_income:
            goods_income[good] += quantity*price
        else:
            goods_income[good] = quantity*price
    return goods_income
# Викликаємо функцію для обчислення доходу
result_income = calculate_income(sells_inventory)
print ("\nДохід кожного продукта", result_income)
''' Створення списку продуктів з доходом більше 1000
.items() - це метод словників у Python, який дозволяє перебрати ВСІ ключі та значення одночасно.# продукт = "ноутбук", дохід = 1500 (перша ітерація)
.append() - метод для ДОДАВАННЯ елементів до списку'''
upper_income_goods = []
for good,income in result_income.items():
    if income > 1000:
        upper_income_goods.append(good)
print("\nВисокодохідні продукти:", upper_income_goods)