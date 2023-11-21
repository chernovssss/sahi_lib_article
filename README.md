# SAHI: Slicing Aided Hyper Inference 
> A lightweight vision library for performing large scale object detection & instance segmentation  
> Оригинальная статья: [Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection](https://ieeexplore.ieee.org/document/9897990)  
> Официальный репозиторий: [github](https://github.com/obss/sahi)
  

Современные детекторы плохо справляются с детекцией небольших объектов. 
    Что бы решить эту проблему был разработан SAHI.  

SAHI "с нахлёстом" разрезает изображение на несколько 
    изображений поменьше. На них уже производится инференс, 
    а резутьтаты склеиваются обратно в исходное изображение.

Гифка с демонстарцией:  
<img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif">  

SAHI может быть использован поверх любого детектора.  
Имеются [CLI команды](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage)  
Интеграция с [roboflow](https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/)  



## Проблема Small Object Detection  

Объекты относительно небольшого размера -
    это объекты, которые обычно имеют ограниченную пространственную 
    протяженность и низкий охват пикселей, и их может быть сложно 
    обнаружить из-за их небольшого внешнего вида и низкого отношения 
    сигнал/шум.

Детектить их сложно потому что: 
1. Limited Receptive Field
2. Limited Spatial Resolution
3. Limited Contextual Information
4. Class Imbalance
5. Feature Representation
6. Scale Variation
7. Training Data Bias  

[источник_1][1] [источник_2][2]  

Можно обратить внимание на репозиторий 
[Awesome Tiny Object Detection](https://github.com/kuanhungchen/awesome-tiny-object-detection).
Там список статей в которых люди занимаются детекцией/сегментацией/локализацией чего-то мальенького.

или заглянуть сюда и посмотреть как применяется SAHI и какие результаты даёт:   
[List of publications that cite SAHI (currently 100+)](https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=14065474760484865747&scipsc=&q=&scisbd=1)

## Использование

Весь важный код в [slicing.py](https://github.com/obss/sahi/blob/main/sahi/slicing.py) 

get_sliced_prediction - Функция для разрезки изображения -> 
    получение прогноза для каждого среза -> объединение прогнозов в полное изображение.  

Простой вариант использования:
```python
result = get_sliced_prediction(
    "img_path" | np.ndarray,
    detection_model, # DetectionModel
    slice_height = 256, # Height of each slice
    slice_width = 256, # Width of each slice
    overlap_height_ratio = 0.2, # Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels). 
    overlap_width_ratio = 0.2 # Fractional overlap in width of each window
)
```
По тому же интерфейсу можно резать картинки "на ходу" с помощью sahi.slicing.slice_image 
    или с помощью sahi.slicing.slice_coco нарезать из деректории для дальнего использования. 

Посмотреть/позапускать нарезку можно тут: [slicing.ipynb](https://github.com/obss/sahi/blob/main/demo/slicing.ipynb)

или можно заглянуть в официальные туториалы:

- `YOLOX` + `SAHI` demo: <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="sahi-yolox"></a> (RECOMMENDED)
- `YOLOv5` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolov5"></a>
- `MMDetection` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-mmdetection"></a>
- `Detectron2` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-detectron2"></a>
- `HuggingFace` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-huggingface"></a> 
- `TorchVision` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-torchvision"></a> 


## CLI Commands

Имеются также [CLI команды](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage).

| Command                                                                                               | Description                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [predict](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage)                   | perform sliced/standard video/image prediction using any [yolov5](https://github.com/ultralytics/yolov5)/[mmdet](https://github.com/open-mmlab/mmdetection)/[detectron2](https://github.com/facebookresearch/detectron2)/[huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads) model                                                                |
| [predict-fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-fiftyone-command-usage) | perform sliced/standard prediction using any [yolov5](https://github.com/ultralytics/yolov5)/[mmdet](https://github.com/open-mmlab/mmdetection)/[detectron2](https://github.com/facebookresearch/detectron2)/[huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads) model and explore results in [fiftyone app](https://github.com/voxel51/fiftyone) |
| [coco slice](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-slice-command-usage)             | automatically slice COCO annotation and image files                                                                                                                                                                                                                                                                                                                                     |
| [coco fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-fiftyone-command-usage)       | explore multiple prediction results on your COCO dataset with [fiftyone ui](https://github.com/voxel51/fiftyone) ordered by number of misdetections                                                                                                                                                                                                                                     |
| [coco evaluate](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-evaluate-command-usage)       | evaluate classwise COCO AP and AR for given predictions and ground truth                                                                                                                                                                                                                                                                                                                |
| [coco analyse](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-analyse-command-usage)         | calculate and export many error analysis plots                                                                                                                                                                                                                                                                                                                                          |
| [coco yolov5](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-yolov5-command-usage)           | automatically convert any COCO dataset to [yolov5](https://github.com/ultralytics/yolov5) format                                                                                                                                                                                                                                                                                        |

[1]: https://learnopencv.com/slicing-aided-hyper-inference/ 
[2]: https://encord.com/blog/slicing-aided-hyper-inference-explained/
