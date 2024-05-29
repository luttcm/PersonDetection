# PersonDetection
Этот проект для решения задачи детекции людей на видео с помощью библиотеки OpenCV и YOLOv3. Код обрабатывает каждый кадр видео, обнаруживает объекты и фильтрует их, чтобы отображать только рамки вокруг людей.



# Установка

**Клонируйте репозиторий:**
```
git clone https://github.com/luttcm/PersonDetection.git
cd PersonDetection
```
**Создайте виртуальное окружение и активируйте его:**
```
python3 -m venv venv
source venv/bin/activate
```

**Скачайте веса и конфигурационные файлы YOLOv3:**

Чтобы скачать веса выполните этот код:
```
wget https://pjreddie.com/media/files/yolov3.weights
```
**Поместите скачанные файлы в директорию проекта:**
```
PersonDetection/
├── yolov3.weights
├── yolov3.cfg
├── coco.names
└── PersonDetection.ipynb
```

# Объяснение кода
## Импорт библиотек
```
import cv2
import numpy as np
import time
```
Импорт необходимых библиотек для обработки видео, численных операций и измерения времени.

## Инициализация захвата видео
```
video = cv2.VideoCapture('/path/to/your/video.mp4')
Создание объекта VideoCapture для чтения входного видео.
```

## Загрузка сети YOLO
```
network = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    labels = f.read().strip().split('\n')
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
```
Загрузка сети YOLO с весами и конфигурационным файлом, а также загрузка меток классов.


## Установка параметров
```
probability_minimum = 0.5
threshold = 0.3
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
```
Определение параметров для минимальной вероятности, порога для подавления немаксимумов и генерация случайных цветов для каждого класса.

## Обработка кадров видео
```
while True:
    ret, frame = video.read()
    if not ret:
        break
    if w is None or h is None:
        h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)
    
    # Обработка обнаружений и рисование ограничивающих рамок...
```
Чтение кадров из видео, предобработка их для создания блоба и выполнение прямого прохода через сеть для получения обнаружений.

## Рисование ограничивающих рамок
```
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
if len(results) > 0:
    for i in results.flatten():
        if classIDs[i] == 0:  # Обработка только класса "человек" (ID класса 0)
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = colours[classIDs[i]].tolist()
            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)
            text_box_current = f'{labels[0]}: {confidences[i]:.4f}'
            cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
```
Фильтрация обнаружений для включения только людей и рисование ограничивающих рамок вокруг них с указанием вероятностей.

## Сохранение выходного видео
```
if writer is None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('output_persons.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
writer.write(frame)
```
Инициализация объекта VideoWriter для сохранения выходного видео и запись обработанных кадров в него.

## Освобождение ресурсов
```
video.release()
writer.release()
print(f'Общее количество кадров: {f}')
print(f'Общее время: {t:.5f} секунд')
print(f'FPS: {round(f / t, 1)}')
```
Освобождение ресурсов захвата и записи видео, а также вывод общего количества обработанных кадров, общего времени и кадров в секунду (FPS).

# Результаты

Выходное видео будет сохранено как submit.mp4 в директории проекта с нарисованными ограничивающими рамками вокруг обнаруженных людей.
