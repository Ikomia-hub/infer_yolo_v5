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

import os
from ikomia import core, dataprocess
from ikomia.utils import qtconversion, pyqtutils
from infer_yolo_v5.infer_yolo_v5_process import InferYoloV5Param
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class InferYoloV5Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYoloV5Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model")
        self.combo_model.addItem("yolov5n")
        self.combo_model.addItem("yolov5s")
        self.combo_model.addItem("yolov5m")
        self.combo_model.addItem("yolov5l")
        self.combo_model.addItem("yolov5x")
        self.combo_model.setCurrentText(self.parameters.model_name)

        self.combo_dataset = pyqtutils.append_combo(self.grid_layout, "Trained on")
        self.combo_dataset.addItem("COCO")
        self.combo_dataset.addItem("Custom")
        self.combo_dataset.setCurrentIndex(0 if self.parameters.dataset == "COCO" else 1)
        self.combo_dataset.currentIndexChanged.connect(self.on_combo_dataset_changed)

        self.label_model_weight = QLabel("Model weight file")
        self.browse_model = pyqtutils.BrowseFileWidget(path=self.parameters.model_weight_file,
                                                       tooltip="Select file",
                                                       mode=QFileDialog.ExistingFile)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_model_weight, row, 0)
        self.grid_layout.addWidget(self.browse_model, row, 1)
        self.label_model_weight.setVisible(False if self.parameters.dataset == "COCO" else True)
        self.browse_model.setVisible(False if self.parameters.dataset == "COCO" else True)

        self.spin_size = pyqtutils.append_spin(self.grid_layout,
                                               "Input size",
                                               self.parameters.input_size)

        self.spin_confidence = pyqtutils.append_double_spin(self.grid_layout,
                                                            "Confidence",
                                                            self.parameters.conf_thres,
                                                            step=0.05,
                                                            decimals=2)

        self.spin_iou = pyqtutils.append_double_spin(self.grid_layout,
                                                     "IOU threshold",
                                                     self.parameters.iou_thres,
                                                     step=0.05,
                                                     decimals=2)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_combo_dataset_changed(self, index):
        if self.combo_dataset.itemText(index) == "COCO":
            self.label_model_weight.setVisible(False)
            self.browse_model.setVisible(False)
            self.browse_model.set_path(self.combo_model.currentText() + ".pt")
        else:
            self.label_model_weight.setVisible(True)
            self.browse_model.setVisible(True)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.dataset = self.combo_dataset.currentText()

        if self.combo_dataset.currentText() == "COCO":
            models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
            self.parameters.model_weight_file = models_folder + os.sep + self.parameters.model_name + ".pt"
        else:
            self.parameters.model_weight_file = self.browse_model.path

        self.parameters.input_size = self.spin_size.value()
        self.parameters.conf_thres = self.spin_confidence.value()
        self.parameters.iou_thres = self.spin_iou.value()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferYoloV5WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_yolo_v5"

    def create(self, param):
        # Create widget object
        return InferYoloV5Widget(param, None)
