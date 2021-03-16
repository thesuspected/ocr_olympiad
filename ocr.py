from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import sys
import json

# Возвращаемый словарь
result_dict = {
    # количество красных участков (штампы, печати и т.д.) на скане
    'red_areas_count': int,
    # количество синих областей (подписи, печати, штампы) на скане
    'blue_areas_count': int,
    # текст главного заголовка страницы или ""
    'text_main_title': str,
    # текстовый блок параграфа страницы, только первые 10 слов, или ""
    'text_block': str,
    # уникальное количество ячеек (сумма количеств ячеек одной или более таблиц)
    'table_cells_count': int,
}


def resize_image(percent, image):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def calc_red_areas_count(image: Image):
    # количество красных участков (штампы, печати и т.д.) на скане
    object_count = 0

    # преобразуем в hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определяем границы поиска по цвету
    red_low = np.array([160, 90, 90])
    red_high = np.array([179, 255, 255])

    # Применяем цветовую маску
    mask = cv2.inRange(hsv_image, red_low, red_high)

    # Объединяем контуры в объекты
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    objects = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(
        objects.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проходимся по контурам
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            object_count += 1

    # отображаем контуры поверх изображения
    cv2.drawContours(image, contours, -1, (255, 0, 0),
                     3, cv2.LINE_AA, hierarchy, 1)

    # Накладываем текст
    cv2.putText(objects, 'red: ' + str(object_count), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Задаем параметр resize для окна
    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    cv2.namedWindow("objects", cv2.WINDOW_NORMAL)

    # Выводим итоговое изображение в окно
    cv2.imshow('contours', resize_image(45, image))
    cv2.imshow('objects', resize_image(45, objects))

    # Записываем количество объектов
    result_dict['red_areas_count'] = object_count

    cv2.waitKey()
    cv2.destroyAllWindows()
    pass


def calc_blue_areas_count(image: Image):
    # количество синих областей (подписи, печати, штампы) на скане
    object_count = 0

    # преобразуем в hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определяем границы поиска по цвету
    blue_low = np.array([85, 80, 80])
    blue_high = np.array([159, 255, 255])

    # Применяем цветовую маску
    mask = cv2.inRange(hsv_image, blue_low, blue_high)

    # Объединяем контуры в объекты
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    objects = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(
        objects.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проходимся по контурам
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            object_count += 1

    # отображаем контуры поверх изображения
    cv2.drawContours(image, contours, -1, (255, 0, 0),
                     3, cv2.LINE_AA, hierarchy, 1)

    # Накладываем текст
    cv2.putText(objects, 'blue: ' + str(object_count), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Задаем параметр resize для окна
    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    cv2.namedWindow("objects", cv2.WINDOW_NORMAL)

    # Выводим итоговое изображение в окно
    cv2.imshow('contours', resize_image(45, image))
    cv2.imshow('objects', resize_image(45, objects))

    # Записываем количество объектов
    result_dict['blue_areas_count'] = object_count

    cv2.waitKey()
    cv2.destroyAllWindows()
    pass


def get_text_main_title():
    # текст главного заголовка страницы или ""
    pass


def get_first_text_block():
    # текстовый блок параграфа страницы, только первые 10 слов, или ""
    pass


def calc_table_cells_count():
    # уникальное количество ячеек (сумма количеств ячеек одной или более таблиц)
    pass


def extract_doc_features(filepath: str) -> dict:
    """
    Функция, которая будет вызвана для получения признаков документа, для которого
   задан:
    :param filepath: абсолютный путь до тестового файла на локальном компьютере (строго
   pdf или png).

    :return: возвращаемый словарь, совпадающий по составу и написанию ключей условию
   задачи
    """
    # Загрузить образ и преобразовать его в оттенки серого
    image = cv2.imread(filepath)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Распознаем красные объекты
    calc_red_areas_count(image.copy())
    # Распознаем синие объекты
    calc_blue_areas_count(image.copy())

    return result_dict


# Вызываем функцию с аргументом абсолютного пути к файлу
if len(sys.argv) == 2:
    # extract_doc_features(sys.argv[1])
    red_eq = 0
    blue_eq = 0

    for number in range(15):
        number += 1
        if number < 10:
            number = '0' + str(number)
        extract_doc_features('tmp/dataset_train/0' + str(number) + '_0e.png')

        with open('tmp/validation_train/0' + str(number) + '_0e.json', 'r') as handle:
            data = json.load(handle)
            # Выводим
            print('file_' + str(number))
            print('red:', data['red_areas_count'],
                  result_dict['red_areas_count'])
            print('blue:', data['blue_areas_count'],
                  result_dict['blue_areas_count'])
            print('')
            # Проверяем совпало ли
            if data['red_areas_count'] == result_dict['red_areas_count']:
                red_eq += 1
            if data['blue_areas_count'] == result_dict['blue_areas_count']:
                blue_eq += 1

    print('red_eq:', str(red_eq) + '/15')
    print('blue_eq:', str(blue_eq) + '/15')
else:
    print('Введите абсолютный путь к файлу')

# image = 'tmp/dataset_train/001_0e.png'

# preprocess = "thresh"

# # загрузить образ и преобразовать его в оттенки серого
# image = cv2.imread(image)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # преобразование в RGB пространство
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # проверьте, следует ли применять пороговое значение для предварительной обработки изображения
# if preprocess == "thresh":
#     gray = cv2.threshold(gray, 0, 255,
#                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# # если нужно медианное размытие, чтобы удалить шум
# elif preprocess == "blur":
#     gray = cv2.medianBlur(gray, 3)

# # сохраним временную картинку в оттенках серого, чтобы можно было применить к ней OCR

# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)

# # загрузка изображения в виде объекта image Pillow, применение OCR, а затем удаление временного файла
# text = pytesseract.image_to_string(Image.open(filename), lang='rus')
# os.remove(filename)
# print(text)
