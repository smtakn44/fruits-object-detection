from ultralytics import YOLO
model=YOLO("best.pt")
model.predict(source='image.jpg', show=True, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2)