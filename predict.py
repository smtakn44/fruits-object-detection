from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source='indir.jpg', show=True, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=True, save_crop=True, line_thickness=2)
