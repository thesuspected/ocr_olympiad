import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# Путь к изображению
file=r'tmp/dataset_train/008_0e.png'
# Параметры картинки
        height, width, sl = image.shape
        center = width/2

#Преобразование изображения в двоичное изображение
thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#Инверсия изображения 
img_bin = 255-img_bin
cv2.imwrite('tmp/cv_inverted.png',img_bin)
# Построение изображения для просмотра вывода
plotting = plt.imshow(img_bin,cmap='gray')
plt.show()

# Длина (ширина) kernel как сотая от общей ширины
kernel_len = np.array(img).shape[1]//100
# Определение вертикального kernel для обнаружения всех вертикальных линий изображения
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Определение горизонтального kernel для обнаружения всех горизонтальных линий изображения
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# Kernel 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

#Обнаружение и сохранение вертикальных линий в jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
cv2.imwrite("tmp/vertical.jpg",vertical_lines)
#Построение сгенерированного изображения
plotting = plt.imshow(image_1,cmap='gray')
plt.show()

#Обнаружение и сохранение горизонтальных линий в jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
cv2.imwrite("tmp/horizontal.jpg",horizontal_lines)
#Построение сгенерированного изображения
plotting = plt.imshow(image_2,cmap='gray')
plt.show()

# Объединение горизонтальных и вертикальных линий в  изображение с "одинаковым весом"
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
#Удаление и сохранение изображения
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("tmp/img_vh.jpg", img_vh)
bitxor = cv2.bitwise_xor(img,img_vh)
bitnot = cv2.bitwise_not(bitxor)
#Построение сгенерированного изображения
plotting = plt.imshow(bitnot,cmap='gray')
plt.show()

# Обнаружение контуров
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):
    # реверснуть и отсортировать по индексу
    reverse = False
    i = 0
    
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    #обработать, если мы сортируем по координате y, а не 
    # по координате x ограничивающего прямоугольника
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # построить список ограничивающих рамок и отсортировать их сверху 
    # вниз 
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # Возвращает список отсортированных контуров и bounding boxes
    return (cnts, boundingBoxes)

# Сортировать все контуры сверху вниз
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

# Создание списка высот для всех обнаруженных ячеек
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

# Получить среднее значение высот
mean = np.mean(heights)

# Создать список для хранения всех ячеек 
box = []
# Получить позицию (x, y), ширину и высоту для каждого контура и показать контур на изображении
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w<1000 and h<500):
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        box.append([x,y,w,h])
        
plotting = plt.imshow(image,cmap='gray')
plt.show()

# Создание кадра данных сгенерированного списка OCR
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print(dataframe)
data = dataframe.style.set_properties(align="left")