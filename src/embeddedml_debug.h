#ifndef EMBEDDEDML_DEBUG_H
#define EMBEDDEDML_DEBUG_H
#include "onnx.pb-c.h"

FILE *fp;
#define DEBUG_PRINT(FMT, ARGS...) do { \
    if (DEBUG) \
        fp = fopen("output.txt", "a"); \
        fprintf(fp, "%s:%d " FMT "\n", __FUNCTION__, __LINE__, ## ARGS); \
    } while (0)

/* Backup
#define DEBUG_PRINT(FMT, ARGS...) do { \
  if (DEBUG) \
      fprintf(stderr, "%s:%d " FMT "\n", __FUNCTION__, __LINE__, ## ARGS); \
    } while (0)*/

void Debug_PrintArray(float *array, int m, int n);
void Debug_PrintModelInformation(Onnx__ModelProto *model);
void Debug_PrintTensorProto(Onnx__TensorProto *tp);

#endif
