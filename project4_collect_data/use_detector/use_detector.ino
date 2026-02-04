/*
 * -----------------------------------------------------------------------------
 * Project 4: Posture Detection (Serial Version - Based on Course Template)
 * Author: Shu-Wen yeh
 * Date: Nov 14, 2025
 *
 * Description:
 * 1. BASED 100% ON YOUR "model_deployment_layout.ino" template.
 * 2. USES "AllOpsResolver" and "MicroErrorReporter" as requested.
 * 3. REMOVED ALL BLE code to save RAM (this is the only way).
 * 4. ADDED IMU code for data collection.
 * 5. ADDED Serial commands ('1','2','3') for base-station control.
 * 6. ADDED your 76.2% int8 model logic.
 * -----------------------------------------------------------------------------
 */

// --- 1. Includes (Based on your template + IMU) ---
#include <Arduino_BMI270_BMM150.h> // <--- ADDED
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
// #include <tensorflow/lite/version.h>
#include "model.h"

// Constants for ACCEL
const float NORM_MEAN_ACCEL[] = { 0.356837f, -0.049351f, 0.022777f };
const float NORM_SCALE_ACCEL[] = { 0.614898f, 0.556168f, 0.759103f };
// Constants for GYRO
const float NORM_MEAN_GYRO[] = { 0.792414f, 0.271208f, -0.231060f };
const float NORM_SCALE_GYRO[] = { 57.045824f, 122.999524f, 98.049251f };
// Constants for MAG
const float NORM_MEAN_MAG[] = { 7.071025f, -19.211967f, -5.879854f };
const float NORM_SCALE_MAG[] = { 23.375511f, 22.936115f, 25.879325f };

// --- 3. TFLite Globals (From your template) ---
tflite::ErrorReporter* error_reporter = nullptr; // <--- MODIFIED (to match modern examples)
tflite::AllOpsResolver resolver; // <--- Your template uses this
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// --- 4. Tensor Arena (MODIFIED to 15KB) ---
// Your 11KB model needs more than 8KB. 15KB is safe.
constexpr int kTensorArenaSize = 15 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// --- 5. Project Globals (ADDED) ---
const int kWindowSize = 100;
const int kInputChannels = 3;
const int kNumClasses = 5;
const char* CLASSES[] = {"sitting", "supine", "prone", "side", "unknown"};
float sampleRate = 50.0;
long interval = 1000.0 / sampleRate;
long lastMillis = 0;
float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;


void setup() {
  Serial.begin(115200);
  delay(1000); // Wait for Serial Monitor
  Serial.println("--- Posture Project (Serial Version) Booting ---");

  // 1. TFLite Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  Serial.println("Error Reporter OK.");

  // 2. IMU (ADDED)
  if (!IMU.begin()) {
    error_reporter->Report("Failed to initialize IMU!");
    Serial.println("!!! 錯誤: Failed to initialize IMU! !!!");
    while(1);
  }
  Serial.println("IMU OK.");

  // 3. TFLite Model (From your template, but fixed var name)
  model = tflite::GetModel(model_tflite); // <--- FIXED (was 'model')
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch!");
    Serial.println("!!! 錯誤: Model schema mismatch! !!!");
    while (1);
  }
  Serial.println("Model Get OK.");

  // 4. TFLite Interpreter (From your template)
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  Serial.println("Interpreter OK.");

  // 5. Allocate Tensors (From your template)
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    Serial.println("!!! 錯誤: AllocateTensors() failed !!!");
    return;
  }
  Serial.println("TFLite AllocateTensors() OK.");

  // 6. Get Tensors (From your template)
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  Serial.println("Get Tensors OK.");

  // 7. Get Quantization Params (ADDED)
  input_scale = model_input->params.scale;
  input_zero_point = model_input->params.zero_point;
  output_scale = model_output->params.scale;
  output_zero_point = model_output->params.zero_point;
  Serial.println("Get Quantization OK.");

  Serial.println("\n--- Arduino Ready (Serial) ---");
  Serial.println("Waiting for command (1=Accel, 2=Gyro, 3=Mag)...");
}

void loop() {
  // 監聽 USB 序列埠 (Replaced BLE)
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == '1' || cmd == '2' || cmd == '3') {
      Serial.print("Received command via Serial: ");
      Serial.println(cmd);
      perform_inference(cmd);
    }
  }
}

// This function is your custom logic, replacing the simple loop()
void perform_inference(char sensor_type) {
  
  float ax, ay, az, gx, gy, gz, mx, my, mz;
  
  // 1. 收集 2 秒 (100 筆) 數據
  for (int i = 0; i < kWindowSize; i++) {
    while (millis() - lastMillis < interval);
    lastMillis = millis();
    if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable() || !IMU.magneticFieldAvailable()) {
      i--; continue;
    }
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    IMU.readMagneticField(mx, my, mz);
    
    float x, y, z;
    float x_norm, y_norm, z_norm;

    // 2. 根據 'sensor_type' 選擇 *正確的* 常數來正規化
    if (sensor_type == '1') {
      x = ax; y = ay; z = az;
      x_norm = (x - NORM_MEAN_ACCEL[0]) / NORM_SCALE_ACCEL[0];
      y_norm = (y - NORM_MEAN_ACCEL[1]) / NORM_SCALE_ACCEL[1];
      z_norm = (z - NORM_MEAN_ACCEL[2]) / NORM_SCALE_ACCEL[2]; 
    } else if (sensor_type == '2') {
      x = gx; y = gy; z = gz;
      x_norm = (x - NORM_MEAN_GYRO[0]) / NORM_SCALE_GYRO[0];
      y_norm = (y - NORM_MEAN_GYRO[1]) / NORM_SCALE_GYRO[1];
      z_norm = (z - NORM_MEAN_GYRO[2]) / NORM_SCALE_GYRO[2]; 
    } else { // sensor_type == '3'
      x = mx; y = my; z = mz;
      x_norm = (x - NORM_MEAN_MAG[0]) / NORM_SCALE_MAG[0];
      y_norm = (y - NORM_MEAN_MAG[1]) / NORM_SCALE_MAG[1];
      z_norm = (z - NORM_MEAN_MAG[2]) / NORM_SCALE_MAG[2]; 
    }

    // 3. 量化 (float32 -> int8)
    model_input->data.int8[i * kInputChannels + 0] = (int8_t) round((x_norm / input_scale) + input_zero_point);
    model_input->data.int8[i * kInputChannels + 1] = (int8_t) round((y_norm / input_scale) + input_zero_point);
    model_input->data.int8[i * kInputChannels + 2] = (int8_t) round((z_norm / input_scale) + input_zero_point);
  }
  
  // 4. 執行 TFLite 推論
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed!");
    return;
  }

  // 5. 處理輸出
  int predicted_class_index = -1;
  int8_t max_score_int8 = -128;
  for (int i = 0; i < kNumClasses; i++) {
    int8_t current_score_int8 = model_output->data.int8[i];
    if (current_score_int8 > max_score_int8) {
      max_score_int8 = current_score_int8;
      predicted_class_index = i;
    }
  }

  // 6. 將預測結果 (人類可讀的) 發送回基地台
  Serial.print("Prediction: ");
  Serial.println(CLASSES[predicted_class_index]);
}