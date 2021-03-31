from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import sys
from pdf2image import convert_from_path

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


def isolate_lines(src, structuring_element, kernel):
    # Ф-я изолирования линий таблицы
    cv2.erode(src, structuring_element, src, (-1, -1))
    cv2.dilate(src, structuring_element, src, (-1, -1))
    # Дополнительно удлиняем линии
    cv2.dilate(src, kernel, src, (-1, -1))


def calc_red_areas_count(image: Image, isCalc=0):
    # количество красных участков (штампы, печати и т.д.) на скане
    object_count = 0
    img_copy = image.copy()
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
            # Закрашиваем печати на копии картинки
            ex = 7  # Увеличение области
            con = np.array([[x-ex, y-ex], [x-ex, y+h+ex],
                            [x+w+ex, y+h+ex], [x+w+ex, y-ex]])
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.fillPoly(img_copy, pts=[con], color=(255, 255, 255))
            pass

    # отображаем контуры поверх изображения
    cv2.drawContours(image, contours, -1, (255, 0, 0),
                     3, cv2.LINE_AA, hierarchy, 1)

    # Накладываем текст
    cv2.putText(closed, 'red: ' + str(object_count), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Записываем количество объектов
    if(isCalc):
        result_dict['red_areas_count'] = object_count

    return img_copy


def calc_blue_areas_count(image: Image, isCalc=0):
    # количество синих областей (подписи, печати, штампы) на скане
    object_count = 0
    img_copy = image.copy()
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
            # Закрашиваем печати на копии картинки
            ex = 7  # Увеличение области
            con = np.array([[x-ex, y-ex], [x-ex, y+h+ex],
                            [x+w+ex, y+h+ex], [x+w+ex, y-ex]])
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.fillPoly(img_copy, pts=[con], color=(255, 255, 255))
            pass

    # отображаем контуры поверх изображения
    cv2.drawContours(image, contours, -1, (255, 0, 0),
                     3, cv2.LINE_AA, hierarchy, 1)

    # Накладываем текст
    cv2.putText(closed, 'blue: ' + str(object_count), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Записываем количество объектов
    if isCalc:
        result_dict['blue_areas_count'] = object_count

    return img_copy


def get_text_main_title(image: Image, isCalc=0):
    # Параметры картинки
    img_copy = image.copy()
    height, width, sl = image.shape
    center = width / 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Убрать внешнюю рамку
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    calc_areas = 0
    # Определить размер области. Если есть внешний контур, то закрасить
    for c in cnts:
        calc_areas += 1
    if (calc_areas <= 2):
        cv2.floodFill(thresh, None, (0, 0), 255)
        cv2.floodFill(thresh, None, (0, 0), 0)
        pass

    # Создание прямоугольных структурных элементов, их объединение
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    # Ищем контуры
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Сортируем контуры по высоте, сверху вниз (по Y)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][1]))
    # Поиск прямоугольника, приближенного к центру с минимальным Y
    rect_points = [0, height, width, 0]  # макс координаты
    point_accuracy = 500  # Допустимая погрешность в пикс.
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # разница расстояний до краев листа
        middle = abs(x - (width - (x + w))) <= point_accuracy
        h_check = h > 25
        if y < rect_points[1] and middle and h_check:
            rect_points = [x, y, x + w, y + h]
            # Закрасить заголовок на копии картинки
            contours = np.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]])
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.fillPoly(img_copy, pts=[contours], color=(255, 255, 255))
            pass

    text = ""
    text = pytesseract.image_to_string(thresh[rect_points[1]:rect_points[3], rect_points[0] + 20:rect_points[2] - 20],
                                       config='--psm 13', lang='rus')
    # Записываем результат
    if isCalc:
        result_dict['text_main_title'] = text[:-2]

    return img_copy


def get_first_text_block(image: Image, isCalc=0):
    # текстовый блок параграфа страницы, только первые 10 слов, или ""
    img_copy = image.copy()
    height, width, sl = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = gray.copy()

    thresh1, img_bin = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # was 128
    img_bin = 255-img_bin
    kernel_len = np.array(image).shape[1] // 89  # Настройка длины линии
    # Находим горизонтальные линии
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    # Находим контуры
    contours_del, hierarchy = cv2.findContours(
        horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for c in contours_del:
        x, y, w, h = cv2.boundingRect(c)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][1]))
    points_max = [width/2, 9999, width/2, 0, 9999]
    i = 0
    for a in cnts:
        x1, y1, w1, h1 = cv2.boundingRect(a)
        for b in contours_del:
            x2, y2, w2, h2 = cv2.boundingRect(b)
            if x2 > x1 and x2 < (x1+w1) and y2 > y1 and y2 < (y1+h1):
                contours = np.array(
                    [[x1, y1], [x1, y1+h1], [x1+w1, y1+h1], [x1+w1, y1]])
                mask = np.zeros((h1 + 2, w1 + 2), np.uint8)
                cv2.fillPoly(thresh, pts=[contours], color=(0, 0, 0))
        if (w1-(points_max[2]-points_max[0])) > 50:
            points_max = [x1, y1, (x1 + w1), (y1 + h1)]
            i = 1

    if(i == 1):
        text = pytesseract.image_to_string(
            thresh[points_max[1]:points_max[3], points_max[0]+20:points_max[2]]-20, lang='rus')
    else:
        text = ""

    res = text.split(' ')
    t = 0
    text = ""
    try:
        for t in range(10):
            text = text + " " + res[t]
    except IndexError:
        text = ""

    # Записываем результат
    if isCalc:
        result_dict['text_block'] = text

    pass


def calc_table_cells_count(image: Image, isCalc=0):
    # Переводим в серые цвета
    img_copy = image.copy()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Задаем параметры для фильтрации
    MAX_THRESHOLD_VALUE = 255
    BLOCK_SIZE = 15
    THRESHOLD_CONSTANT = 0

    # Фильтруем изображение
    filtered = cv2.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)

    # Задаем масштаб линии
    SCALE = 75

    horizontal = filtered.copy()
    vertical = filtered.copy()

    # Проходимся по горизонтальным линиям
    horizontal_size = int(horizontal.shape[1] / SCALE)
    horizontal_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (9, 1))
    isolate_lines(horizontal, horizontal_structure, horizontal_kernel)

    # Проходимся по вертикальным линиям
    vertical_size = int(vertical.shape[0] / SCALE)
    vertical_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_size))
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 9))
    isolate_lines(vertical, vertical_structure, vertical_kernel)

    # Создаем маску
    mask = horizontal + vertical
    (contours, hierarchy) = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Находим ячейки
    intersections = cv2.bitwise_and(horizontal, vertical)

    cell_count = 0

    for j, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 30 and h > 30 and hierarchy[0][j][3] != -1):
            cell_count += 1

            con1 = np.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]])
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.fillPoly(img_copy, pts=[con1], color=(255, 255, 255))

    # Записываем количество объектов
    if isCalc:
        result_dict['table_cells_count'] = cell_count

    return img_copy


def extract_doc_features(filepath: str) -> dict:
    # Проверяем формат документа
    if not filepath.endswith(".pdf") and not filepath.endswith(".png"):
        print("Для запуска программы используйте только файлы формата pdf или png.")
        sys.exit(1)

    # Если pdf, конвертируем
    if filepath.endswith(".pdf"):
        ext_img = convert_from_path(filepath)[0]
    else:
        ext_img = Image.open(filepath)

    # Сохраняем изображение
    ext_img.save("target.png", "PNG")
    img = cv2.imread("target.png")

    # Заголовок текста
    get_text_main_title(calc_blue_areas_count(
        calc_red_areas_count(img.copy())), 1)
    # Синие области
    calc_blue_areas_count(img.copy(), 1)
    # Красные области
    calc_red_areas_count(img.copy(), 1)
    # Первые 10 слов параграфа
    get_first_text_block(calc_table_cells_count(get_text_main_title(
        calc_blue_areas_count(calc_red_areas_count(img.copy())))), 1)
    # Кол-во ячеек
    calc_table_cells_count(img.copy(), 1)

    return result_dict
