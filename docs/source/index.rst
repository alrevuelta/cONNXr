:orphan:

.. meta::
   :description lang=en: Machine learning runtime written in pure C99 with no dependencies to run inference on models using the open standard onnx.

.. note::
   This is project is in a very early stage and we are looking for contributors.

======
cONNXr
======
A onnx runtime written in pure C99 with zero dependencies focused on embedded devices. Run inference on your machine learning models no matter which framework you train it with and no matter the device that you use. This is the perfect way to go in old hardware that doesn't support fancy modern C or C++.

=======================
Out of the box example
=======================

Estimate the number written in an image using the MNIST model. The input image is stored in `input_0.pb` and the MNIST model is stored in onnx format in `model.onnx`.

First compile the project.

.. code-block:: shell

   make all

And run inference on the input using the model.

.. code-block:: shell

   build/connxr test/mnist/model.onnx test/mnist/test_data_set_0/input_0.pb

.. toctree::
	:maxdepth: 2
	:hidden:
	:caption: Documentation

	documentation
..
  More sections to be added

.. toctree::
	:maxdepth: 2
	:hidden:
	:caption: Contributing

	contributing

.. toctree::
	:maxdepth: 2
	:hidden:
	:caption: Testing

	testing

.. toctree::
	:maxdepth: 2
	:hidden:
	:caption: Operator Status
	
	operator_status