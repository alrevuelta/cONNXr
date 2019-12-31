#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*#include "../src/pb/onnx.pb-c.h"*/
/* This example is using nanopb */
#include "../src/pb/nanopb/onnx.pb.h"
#include "../src/pb/nanopb/pb.h" /* is this needed?*/
#include "../src/pb/nanopb/pb_common.h"
#include "../src/pb/nanopb/pb_decode.h"
#include "../src/pb/nanopb/pb_encode.h" /* Not needed ?*/

#include "../src/embeddedml_debug.h"
#include "../src/embeddedml_inference.h"

bool decode_node(pb_istream_t *istream, const pb_field_t *field, void **arg)
{
    IntList * dest = (IntList*)(*arg);

    // decode single number
    int64_t number;
    if (!pb_decode_svarint(istream, &number))
    {
        const char * error = PB_GET_ERROR(istream);
        printf("SimpleMessage_decode_single_number error: %s", error);
        return false;
    }

    // add to destination list
    IntList_add_number(dest, (int32_t)number);
    return true;
}

int main(int argc, char **argv)
{
  onnx_ModelProto msg = {};

  FILE *fl = fopen("mnist/model.onnx", "r");
  if (fl == NULL){
    TRACE_LEVEL0("File was not opened");
  }
  fseek(fl, 0, SEEK_END);
  long len = ftell(fl);
  uint8_t *buffer = malloc(len);
  fseek(fl, 0, SEEK_SET);
  fread(buffer, 1, len, fl);
  fclose(fl);
  TRACE_LEVEL0("length of file is %ld", len);

  pb_istream_t stream;
  stream = pb_istream_from_buffer(buffer, len);
  bool res = pb_decode(&stream, onnx_ModelProto_fields, &msg);
  if(!res){
    printf("something went wrong\n");
  }

  printf("producer_name = %s\n", msg.producer_name);
  printf("producer_name = %s\n", msg.graph.node.arg);


}
