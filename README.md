# ProStack-AfidPython
Ссылка на репозиторий програмного кода проекта Afid на языке MATLAB: https://github.com/ellispatrick/AFidMatlab<br>
Здесь описаны шаги для внедрение AFid - инстурмент удаления автофлуоресценции в систему обработки изображений ProStack.<br>

## Параметры для этой программы: <br>
python afid.py --noise_del_cf 0.7 --kAuto 0 --k 6 --min_area 7 --corr 0.3 --trace_sensitivity 20 input_image_0_chanel.tif,input_image_1_chanel.tif output_image_0_chanel.tif,output_image_1_chanel.tif<br>
1. noise_del_cf - коэффициент удаления шума.
2. kAuto - 1 если автоматически подобрать число кластеров, иначе 0.
3. k - число кластеров.
4. min_area - минимальная плоадь связных объектов.
5. corr - пороговый коэффициент корреляции Пирсона.
<br>Вариации алгоритма:<br>
Первая вариация наиболее простая :объекты считаются автофлуоресцентными если значение межканальных коэффициентов корреляции Пирсона превышает пороговое значение.
Для этого задайте значение параметра corr, утсановите k = 1 И kAuto = 0<br>
Для второй вариации применяется кластеризация вычисленных характеристик где число кластеров задается вручную. Установите k больше 1 и kAuto = 0 <br>
Для третьей предусмотрено автоматическое определение оптимального числа кластеров с вычислением значений статистик на каждой итерации перебора числа кластеров. 
Установите kAuto = 1<br>
 <br>
Example comand<br>
python afid.py --noise_del_cf 0.7 --kAuto 0 --k 6 --min_area 7 --corr 0.3 --trace_sensitivity 20 C:\vkr-temirgaliev\ProStack-AfidPython\m7\49\m7_0_49.tif,C:\vkr-temirgaliev\ProStack-AfidPython\m7\49\m7_1_49.tif C:\vkr-temirgaliev\ProStack-AfidPython\m7\49\or\im1_res_removed_m7_49.tif,C:\vkr-temirgaliev\ProStack-AfidPython\m7\49\or\im2_res_removed_m7_49.tif

## Создание исполняемого файла
- Установите `pyinstaller`: pip pyinstaller
- Используйте команду в директории исходных файлов: pyinstaller afid.py
- В папке `dist/afid` появится всё необходимое (библиотеки и пакеты) чтобы перенести в директорию `bin/` ProStack и запускать исполняемый файл `afid.exe` 


## Команда для добавления блока отвечающий за этот алгоритм в базу данных простака
Файл с sqlt кодом `afid.sqlt3` расположен в данном репозитории по адресу `sqlt/` <br>
Команды: <br>
- cd /../.bambu/
- sqlite3 "kimono-db.db.en_US.UTF-8"
- cat afid.sqlt3