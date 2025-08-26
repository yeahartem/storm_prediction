import numpy as np

# Укажите путь к файлу, который хотите проверить
file_path = 'data/cmip5_infer/time.npy'

try:
    # Загружаем данные из файла
    data = np.load(file_path)

    # Выводим полезную информацию
    print(f"--- Анализ файла: {file_path} ---")
    print(f"Форма массива (shape): {data.shape}")
    print(f"Тип данных (dtype): {data.dtype}")
    print(f"Минимальное значение: {np.min(data)}")
    print(f"Максимальное значение: {np.max(data)}")

    # Посмотрим на небольшой срез данных (первые 10 значений в первой строке первой "страницы")
    # print(f"Пример данных (срез): {data[0, 0, :10]}")
    print(f"Пример данных (срез): {data[:10]}")

except FileNotFoundError:
    print(f"ОШИБКА: Файл не найден! Убедитесь, что путь '{file_path}' верный.")
except Exception as e:
    print(f"Произошла ошибка: {e}")