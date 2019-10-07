#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void arrayToString(float *array, int *dims, int nDims, char *inputs){
  //Allocate this dynamically
  char buff[1000] = "";
  int totalDim = 0;
  for (int i = 0; i < nDims; i++)
  {
    totalDim += dims[i];
  }

  //printf("[");
  strcat(inputs, "[");
  for (int i = 0; i < totalDim; i++)
  {
    if (i == totalDim - 1)
    {
      sprintf(inputs, "%f", array[i]);
      //printf("%f", array[i]);
    }
    else
    {
      //printf("%f,", array[i]);
      sprintf(inputs, "%f,", array[i]);
    }
  }
  //printf("]");
  strcat(inputs, "]");
}

int main(){

  char cmd[100] = "";
  char inputs[100] = "";

  float array[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  int dims[2] = {4, 4};
  int nDims = 2;
  arrayToString(array, dims, nDims, inputs);

  strcat(cmd, "python3 -c 'import operators_test; operators_test.test_MatMul(");
  strcat(cmd, inputs);
  strcat(cmd, ")' ");

  printf("command is %s\n", cmd);



  //system(cmd);

  return 0;
}



#if 0
#include <Python.h>

int main()
{
  // This is needed to locate the python script
  setenv("PYTHONPATH",".",1);


  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  pName = PyString_FromString("operators_test");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL) {
    pFunc = PyObject_GetAttrString(pModule, "test_MatMul");
    /* pFunc is a new reference */

    if (pFunc && PyCallable_Check(pFunc)) {
      pArgs = PyTuple_New(3);
      for (i = 0; i < 3; ++i) {
        pValue = PyInt_FromLong(3);
        if (!pValue) {
          Py_DECREF(pArgs);
          Py_DECREF(pModule);
          fprintf(stderr, "Cannot convert argument\n");
          return 1;
        }
        /* pValue reference stolen here: */
        PyTuple_SetItem(pArgs, i, pValue);
      }
      pValue = PyObject_CallObject(pFunc, pArgs);
      Py_DECREF(pArgs);
      if (pValue != NULL) {
        printf("Result of call: %ld\n", PyInt_AsLong(pValue));
        Py_DECREF(pValue);
      }
      else {
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr,"Call failed\n");
        return 1;
      }
    }
    else {
      if (PyErr_Occurred())
      PyErr_Print();
      fprintf(stderr, "Cannot find function\n");
    }
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
  }
  else {
    PyErr_Print();
    fprintf(stderr, "Failed to load\n");
    return 1;
  }
  Py_Finalize();
  return 0;
}

#endif
