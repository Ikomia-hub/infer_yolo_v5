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

from ikomia import utils, core, dataprocess
import os
import copy
import sys
import logging
import torch
import numpy as np
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

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = params["model_name"]
        self.dataset = params["dataset"]
        self.input_size = int(params["input_size"])
        self.augment = utils.strtobool(params["augment"])
        self.conf_thres = float(params["conf_thres"])
        self.iou_thres = float(params["iou_thres"])
        self.agnostic_nms = utils.strtobool(params["agnostic_nms"])

        if self.dataset != "COCO":
            self.model_path = params["model_path"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dataset": self.dataset,
            "input_size": str(self.input_size),
            "augment": str(self.augment),
            "conf_thres": str(self.conf_thres),
            "iou_thres": str(self.iou_thres),
            "agnostic_nms": str(self.agnostic_nms)
        }
        return params

# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class InferYoloV5(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        self.model = None
        self.names = None
        self.update = False
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create parameters class
        if param is None:
            self.set_param_object(InferYoloV5Param())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 6

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Step progress bar:
        self.emit_step_progress()

        # Get input :
        img_input = self.get_input(0)
        src_image = img_input.get_image()

        # Make predictions
        with torch.no_grad():
            self.predict(src_image)

        # Call end_task_run to finalize process
        self.end_task_run()

    def predict(self, src_image):
        param = self.get_param_object()

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

            # Get names
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            param.update = False
        else:
            stride = int(self.model.stride.max())  # model stride

        # Resize image
        image = letterbox(src_image, param.input_size, stride)[0]
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        self.emit_step_progress()

        # Run inference
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        self.emit_step_progress()

        # Inference
        pred = self.model(image, augment=param.augment)[0]
        self.emit_step_progress()

        # Apply NMS
        pred = non_max_suppression(
                            pred, param.conf_thres,
                            param.iou_thres,
                            agnostic=param.agnostic_nms
                                )
        self.emit_step_progress()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], src_image.shape).round()

                # Results
                self.set_names(self.names)
                index = 0
                for *xyxy, conf, cls in reversed(det):
                    # Box
                    w = float(xyxy[2] - xyxy[0])
                    h = float(xyxy[3] - xyxy[1])
                    self.add_object(index,
                                    int(cls),
                                    conf.item(),
                                    float(xyxy[0]),
                                    float(xyxy[1]),
                                    w,
                                    h)

                    index += 1

        self.emit_step_progress()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class InferYoloV5Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v5"
        self.info.short_description = "Ultralytics YoloV5 object detection models."
        self.info.description = "This plugin proposes inference on YoloV5 object detection models. " \
                                "Models implementation comes from the Ultralytics team based on " \
                                "PyTorch framework."
        self.info.authors = "Plugin authors"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.3.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Ultralytics"
        self.info.year = 2020
        self.info.license = "GPLv3"
        # Code source repository
        self.info.repository = "https://github.com/ultralytics/yolov5"
        # Keywords used for search
        self.info.keywords = "object, detection, pytorch"

    def create(self, param=None):
        # Create process object
        return InferYoloV5(self.info.name, param)
