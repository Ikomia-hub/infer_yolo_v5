# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import os
import copy
import sys
import logging
import torch
import random
import numpy as np
from distutils.util import strtobool
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox

logger = logging.getLogger()


def init_logging(rank=-1):
    if rank in [-1, 0]:
        logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        info = logging.StreamHandler(sys.stdout)
        info.setLevel(logging.INFO)
        info.setFormatter(formatter)
        logger.addHandler(info)

        err = logging.StreamHandler(sys.stderr)
        err.setLevel(logging.ERROR)
        err.setFormatter(formatter)
        logger.addHandler(err)
    else:
        logging.basicConfig(format="%(message)s", level=logging.WARN)


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class InferYoloV5Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)

        # Create models folder
        models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        os.makedirs(models_folder, exist_ok=True)

        # Place default value initialization here
        self.model_name = "yolov5s"
        self.model_path = models_folder + os.sep + self.model_name + ".pt"
        self.dataset = "COCO"
        self.input_size = 640
        self.augment = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.model_path = param_map["model_path"]
        self.dataset = param_map["dataset"]
        self.input_size = int(param_map["input_size"])
        self.augment = strtobool(param_map["augment"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.agnostic_nms = strtobool(param_map["agnostic_nms"])
        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["model_path"] = self.model_path
        param_map["dataset"] = self.dataset
        param_map["input_size"] = str(self.input_size)
        param_map["augment"] = str(self.augment)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["iou_thres"] = str(self.iou_thres)
        param_map["agnostic_nms"] = str(self.agnostic_nms)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class InferYoloV5(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.model = None
        self.names = None
        self.colors = None
        self.update = False
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CBlobMeasureIO())

        # Create parameters class
        if param is None:
            self.setParam(InferYoloV5Param())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 6

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Step progress bar:
        self.emitStepProgress()

        # Get input :
        img_input = self.getInput(0)
        src_image = img_input.getImage()

        # Make predictions
        with torch.no_grad():
            self.predict(src_image)

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def predict(self, src_image):
        param = self.getParam()

        # Initialize
        init_logging()
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        if self.model is None or param.update:
            self.model = attempt_load(param.model_path, map_location=self.device)  # load FP32 model
            stride = int(self.model.stride.max())  # model stride
            param.input_size = check_img_size(param.input_size, s=stride)  # check img_size
            if half:
                self.model.half()  # to FP16F

            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            param.update = False
        else:
            stride = int(self.model.stride.max())  # model stride

        # Resize image
        image = letterbox(src_image, param.input_size, stride)[0]
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        self.emitStepProgress()

        # Run inference
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        self.emitStepProgress()

        # Inference
        pred = self.model(image, augment=param.augment)[0]
        self.emitStepProgress()

        # Apply NMS
        pred = non_max_suppression(pred, param.conf_thres, param.iou_thres, agnostic=param.agnostic_nms)
        self.emitStepProgress()

        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("YoloV5")
        graphics_output.setImageIndex(0)
        # Init numeric output
        numeric_output = self.getOutput(2)
        numeric_output.clearData()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], src_image.shape).round()

                # Results
                for *xyxy, conf, cls in reversed(det):
                    # Box
                    w = float(xyxy[2] - xyxy[0])
                    h = float(xyxy[3] - xyxy[1])
                    prop_rect = core.GraphicsRectProperty()
                    prop_rect.pen_color = self.colors[int(cls)]
                    graphics_box = graphics_output.addRectangle(float(xyxy[0]), float(xyxy[1]), w, h, prop_rect)
                    graphics_box.setCategory(self.names[int(cls)])
                    # Label
                    name = self.names[int(cls)]
                    prop_text = core.GraphicsTextProperty()
                    prop_text.font_size = 8
                    prop_text.color = self.colors[int(cls)]
                    prop_text.bold = True
                    graphics_output.addText(name, float(xyxy[0]), float(xyxy[1]), prop_text)
                    # object results
                    results = []
                    confidence_data = dataprocess.CObjectMeasure(
                        dataprocess.CMeasure(core.MeasureId.CUSTOM, "Confidence"),
                        conf.item(),
                        graphics_box.getId(),
                        name)
                    box_data = dataprocess.CObjectMeasure(
                        dataprocess.CMeasure(core.MeasureId.BBOX),
                        [float(xyxy[0]), float(xyxy[1]), w, h],
                        graphics_box.getId(),
                        name)
                    results.append(confidence_data)
                    results.append(box_data)
                    numeric_output.addObjectMeasures(results)

        self.emitStepProgress()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class InferYoloV5Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v5"
        self.info.shortDescription = "Ultralytics YoloV5 object detection models."
        self.info.description = "This plugin proposes inference on YoloV5 object detection models. " \
                                "Models implementation comes from the Ultralytics team based on " \
                                "PyTorch framework."
        self.info.authors = "Plugin authors"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Ultralytics"
        self.info.year = 2020
        self.info.license = "GPLv3"
        # Code source repository
        self.info.repository = "https://github.com/ultralytics/yolov5"
        # Keywords used for search
        self.info.keywords = "object,detection,pytorch"

    def create(self, param=None):
        # Create process object
        return InferYoloV5(self.info.name, param)
