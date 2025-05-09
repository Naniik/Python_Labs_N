# 1. Отримати курси евро за попередній тиждень, вивести на екран дату + курс
# 2. З отриманого словника побудувати графк зміни курсу за тиждень
import json
import requests
import matplotlib.pyplot as plt

print("Частина 1")
#URL requests https://bank.gov.ua/NBU_Exchange/exchange_site?start=20250324&end=20250329&valcode=eur&json
response_data = requests.get("https://bank.gov.ua/NBU_Exchange/exchange_site?start=20250324&end=20250430&valcode=eur&json")
print(response_data)
print(response_data.content)
print(response_data.apparent_encoding)
print(response_data.text)

#response_dictionary = json.loads(response_data.content)
#print(response_dictionary)

response_list = json.loads(response_data.content)

'''
for i in response_list:
    print(i)
'''

for item in response_list:
    print(item['exchangedate'], item['rate'])

print("Частина 2")

#Формування списків
exchange_date = []
exchange_rate = []

for item in response_list:
    exchange_date.append(item['exchangedate'])
    exchange_rate.append(item['rate'])

print(exchange_date)
print(exchange_rate)
plt.plot(exchange_date, exchange_rate)
plt.show()