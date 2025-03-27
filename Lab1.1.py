def count_word_frequency(text):
    text_with_words = text.lower().split()
    word_frequency = {}

    for word in text_with_words:

        if word not in word_frequency:
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1

    # Створюємо список слів, що зустрічаються більше 3 разів
    frequent_words_3 = []
    for word, count in word_frequency.items():
        if count > 3:
            frequent_words_3.append(word)

    return word_frequency, frequent_words_3
text = "Кіт сидів на вікні. Кіт дивився на пташку. Кіт стрибнув, але кіт не спіймав пташку."
frequency_dictionary, frequent_words = count_word_frequency(text)
print("Словник частоти слів:")

for word, count in frequency_dictionary.items():
    print(f"Слово '{word}' зустрічається {count} разів")

print("\nСлова, що зустрічаються більше 3 разів:\n", frequent_words)