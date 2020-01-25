#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char small_model_onnx[] = {
  TODO
};
unsigned int small_model_onnx_len = TODO;
Onnx__ModelProto *model = NULL;
void setup() {
  model = onnx__model_proto__unpack(NULL,small_model_onnx_len,small_model_onnx);
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("Hi!");
  Serial.println(model->producer_name);
  Serial.println(model->graph->name);
  Serial.println(model->graph->node[0]->name);

}
