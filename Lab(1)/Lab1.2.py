
inventory = {
    "яблука": 25,
    "банани": 30,
    "молоко": 3,
    "хліб": 10
}
def update_inventory(product, quantity):
    if product in inventory:
        inventory[product] += quantity
    else:
        inventory[product] = quantity

# Функція виведення продуктів з низькою кількістю
def get_low_stock_products():
    # Створюємо список для продуктів з кількістю менше 5
    low_stock = []
    # Перевіряємо кожен продукт в інвентарі
    for product, quantity in inventory.items():
        if quantity < 5:
            print(f"{product}: {quantity} шт.")
            low_stock.append(product)

    return low_stock

print("Поточний інвентар:",inventory)
update_inventory("яблука", -22)
update_inventory("банани", -3)
update_inventory("картопля", 10)
print("\nОновлений інвентар:\n",inventory)
print("\nПродукти з низькою кількістю:\n")
get_low_stock_products()
