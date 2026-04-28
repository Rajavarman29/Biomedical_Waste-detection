from roboflow import Roboflow
rf = Roboflow(api_key="eAaV7TTfnuy50xfqSYGY")
project = rf.workspace("raja-8xfja").project("medical-waste-yggno")
version = project.version(2)
dataset = version.download("yolov8")
            