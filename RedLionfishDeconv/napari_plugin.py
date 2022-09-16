'''
Copyright 2021 Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

#RedLionfish napari plugin

'''
Some help designing this collected from diffrent places

https://napari.org/plugins/stable/hook_specifications.html#gui-hooks
https://github.com/DragaDoncila/example-plugin/blob/main/example_plugin/_dock_widget.py
These examples show adding widget using decorator @magic_factory.
This decorator is poorly documented. I don't know how to use it.


'''

from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation

#As suggested by DragaDoncila
# https://github.com/rosalindfranklininstitute/RedLionfish/pull/5#issuecomment-1239989472
# If napari is not installed
# from typing import TYPE_CHECKING
# print(f"TYPE_CHECKING: {TYPE_CHECKING}")
# if TYPE_CHECKING:
#     from napari.types import LabelsData, ImageData, LayerDataTuple
#     from napari.layers import Image

from napari.types import ImageData, LayerDataTuple
from napari.utils import progress

import RedLionfishDeconv as rl

# import logging
# logging.basicConfig(level=logging.INFO)

#Logging in napari is not working not sure why.
# Maybe napari also uses logging and outputs to different location

# To output to console use print or loguru instead.

import time


''' The parameters after the decorator setup the title and other properties of the widget window.

The parameters in the FUNCTION will set the elements that will be shown, but things like comboboxes
or sliders will have no limits set, this should be done in the decorator.

see also
https://napari.org/guides/stable/magicgui.html

Note that the @magic_factory and @magicgui behave in similar way.
'''

#The widget
@magic_factory (
    call_button="Go" ,
    iterations={'max':16384},
    useGPU={"label": "Use GPU if possible"}, # https://github.com/napari/magicgui/blob/main/examples/change_label.py
    resAsUInt8={"label": "Result as clip-norm integer (uint8)"}
    )
def RedLionfish_widget(
    data: ImageData, #Input is data that can be selected
    psfdata:ImageData,
    iterations=10, #User chooses number of RL iterations
    useGPU = True,
    resAsUInt8=True
    ) -> LayerDataTuple: #Result is a LayerDataTuple, like (data, {dict_properties})

    print(f"iterations = {iterations}")

    ret = None

    datares=None
    if not data is None and not psfdata is None:
        #Print information before calculation
        print(f"data.shape = {data.shape} , type(data) = {type(data)} , data.dtype = {data.dtype}")
        print(f"psfdata.shape = {psfdata.shape} , type(psfdata) = {type(psfdata)} , psfdata.dtype = {psfdata.dtype}")

        pbr=progress(total=iterations)
        #pbr=progress()
        pbr.set_description(f"Calculation progress")

        def callback():
            #print("iteration tick")
            pbr.update(1)
            pbr.refresh()
            #time.sleep(0.1)

        if data.ndim==3 and psfdata.ndim==3: #For now, only 3D is supported
            #logging.basicConfig(level=logging.INFO) #not working

            method = 'cpu' #Default
            if useGPU:
                method='gpu'
            #Run the deconvolution
            datares = rl.doRLDeconvolutionFromNpArrays(data, psfdata, niter= iterations, method = method ,resAsUint8=resAsUInt8 , callbkTickFunc=callback)

            ret = ( datares , { 'name':'RL-deconvolution'})
        else:
            print("Data or PSF is not 3-dimensional.")

        pbr.close()
    return ret



@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #return [test_widget0, test_widget1]
    #return [RedLionfish_widget,test_widget2, test_widget1]
    return RedLionfish_widget