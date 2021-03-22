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

if __name__ == '__main__':
    def nothing(*arg):
        pass


def resize_image(percent, image):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def search_hsv_range():
    cv2.namedWindow("result")  # создаем главное окно
    cv2.namedWindow("settings")  # создаем окно настроек

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
    crange = [0, 0, 0, 0, 0, 0]

    while True:
        img = cv2.imread('tmp/dataset_train/011_0e.png')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # считываем значения бегунков
        h1 = cv2.getTrackbarPos('h1', 'settings')
        s1 = cv2.getTrackbarPos('s1', 'settings')
        v1 = cv2.getTrackbarPos('v1', 'settings')
        h2 = cv2.getTrackbarPos('h2', 'settings')
        s2 = cv2.getTrackbarPos('s2', 'settings')
        v2 = cv2.getTrackbarPos('v2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        thresh = cv2.inRange(hsv, h_min, h_max)

        cv2.imshow('result', resize_image(40, thresh))

        # ESC
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()


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
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проходимся по контурам
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            object_count += 1

    # отображаем контуры поверх изображения
    cv2.drawContours(image, contours, -1, (255, 0, 0),
                     3, cv2.LINE_AA, hierarchy, 1)

    # Накладываем текст
    cv2.putText(closed, 'red: ' + str(object_count), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Задаем параметр resize для окна
    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    cv2.namedWindow("closed", cv2.WINDOW_NORMAL)

    # Выводим итоговое изображение в окно
    cv2.imshow('contours', resize_image(40, image))
    cv2.imshow('closed', resize_image(40, closed))

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
    blue_low = np.array([85, 35, 140])
    blue_high = np.array([159, 255, 255])

    # Применяем цветовую маску
    mask = cv2.inRange(hsv_image, blue_low, blue_high)

    # Объединяем контуры [CLOSED]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Избавляемся от шума [OPENING]
    kernel2 = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    closed = cv2.erode(opening, kernel2, iterations=1)
    closed = cv2.dilate(closed, kernel2, iterations=1)

    # Объединяем ближайшие контуры воедино
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    closed = cv2.dilate(closed, kernel3, iterations=7)
    closed = cv2.erode(closed, kernel3, iterations=7)

    # Ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проходимся по контурам
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Если контур больше 30 пикселей
        if w > 30 and h > 30:
            object_count += 1

    # отображаем контуры поверх изображения
    cv2.drawContours(image, contours, -1, (255, 0, 0),
                     3, cv2.LINE_AA, hierarchy, 1)

    # Накладываем текст
    cv2.putText(closed, 'blue: ' + str(object_count), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Задаем параметр resize для окна
    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    cv2.namedWindow("closed", cv2.WINDOW_NORMAL)

    # Выводим итоговое изображение в окно
    cv2.imshow('contours', resize_image(40, image))
    cv2.imshow('closed', resize_image(40, closed))

    # Записываем количество объектов
    result_dict['blue_areas_count'] = object_count

    cv2.waitKey()
    cv2.destroyAllWindows()
    pass


def get_text_main_title():
    # текст главного заголовка страницы или ""
    image = cv2.imread('tmp/dataset_train/008_0e.png')
    # Параметры картинки
    height, width, sl = image.shape
    center = width/2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Создание прямоугольных структурных элементов, их объединение
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    # Ищем контуры
    rect_points = [0, 9999, 9999, 0, 9999]
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Сортируем контуры по высоте, сверху вниз (по Y)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][1]))
    # Поиск прямоугольника, приближенного к центру с минимальным Y
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        first_point = (center-x) > 0 and (center-x) < (center-rect_points[0])
        second_point = (x+w-center) > 0 and (x+w -
                                             center) < (rect_points[2]-center)
        if y < rect_points[1] and first_point and second_point:
            rect_points = [x, y, x+w, y+h]
            pass
    # Отрисовка прямоугольника
    cv2.rectangle(image, (rect_points[0], rect_points[1]),
                  (rect_points[2], rect_points[3]), (36, 255, 12), 2)
    # text = pytesseract.image_to_string(thresh, lang='rus')

    text = pytesseract.image_to_string(
        thresh[rect_points[1]:rect_points[3], rect_points[0]:rect_points[2]], lang='rus')
    print(text)
    # Изменить размер под экран
    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.imshow('thresh', resize_image(45, thresh))
    # cv2.imshow('dilate', resize_image(45, dilate))
    cv2.imshow('image', resize_image(45, image))
    cv2.waitKey()

    pass


def get_first_text_block():
    # текстовый блок параграфа страницы, только первые 10 слов, или ""
    # https://stackoverflow.com/questions/57249273/how-to-detect-paragraphs-in-a-text-document-image-for-a-non-consistent-text-stru
    # https://muthu.co/all-tesseract-ocr-options/
    # https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document

    # image_path = 'tmp/dataset_train/001_0e.png'
    image = cv2.imread('tmp/dataset_train/012_0e.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Найти максимальный отступ слева, взять верхнее изображение
    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

    # text = pytesseract.image_to_string(thresh, lang='rus')
    # Изменить размер под экран
    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # print(text)

    cv2.imshow('thresh', resize_image(45, thresh))
    cv2.imshow('dilate', resize_image(45, dilate))
    cv2.imshow('image', resize_image(45, image))
    cv2.waitKey()
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
    # Распознаем заголовок
    # get_text_main_title(image.copy())
    return result_dict


# Вызываем функцию с аргументом абсолютного пути к файлу
# Для подсчета печатей/подписей
if sys.argv[1] == "1":
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


# Для вывода заголовка
if sys.argv[1] == "2":
    # get_first_text_block()
    get_text_main_title()

# Для поиска цветового диапазона
if sys.argv[1] == "color":
    search_hsv_range()
