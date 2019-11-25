#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "notimplemented.h"

void operator_notimplemented(const char *operation)
{
  printf("\n\nTODO: INFORM THAT THE OPERATOR IS NOT IMPLEMENTED OR DOESNT EXIST %s\n\n", operation);
}
