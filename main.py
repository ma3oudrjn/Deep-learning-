from ultralytics import YOLO

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

model = YOLO("yolov8x.pt")

#change this path
project_path = "/content/drive/My Drive/carpet"
output_name = "train_output"  # this is the name of the folder where the outputs will be saved

# Define augmentation parameters
augment_params = {
    'flipud': 0.5,      
    'fliplr': 0.5,       
    'mosaic': 1.0,       
    'mixup': 0.5,        
    'degrees': 10.0,      
    'translate': 0.1,     
    'scale': 0.5,         
    'shear': 2.0,       
    'perspective': 0.0, 
    'hsv_h': 0.015,     
    'hsv_s': 0.7,         
    'hsv_v': 0.4        
}

#change this path
model.train(data="/content/drive/My Drive/carpet/data.yaml",
            epochs=1000,
            project=project_path,
            name=output_name,
            **augment_params)

metrics = model.val()
