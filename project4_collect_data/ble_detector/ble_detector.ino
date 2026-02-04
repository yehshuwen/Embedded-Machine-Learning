/*
 * -----------------------------------------------------------------------------
 * Project 4: Posture Detection System (BLE)
 * Author: Shu-Wen Yeh
 *
 * Description:
 * - BLE service.
 * - Phone/tablet sends a command:
 *       '1' = use accelerometer
 *       '2' = use gyroscope
 *       '3' = use magnetometer
 * predict posture 'sitting', 'supine', 'prone', 'side', 'unknown'
 * -----------------------------------------------------------------------------
 */

#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <math.h>
#include "model.h"

// -----------------------------------------------------------------------------
// Normalization constants (computed from offline training data)
// -----------------------------------------------------------------------------

// Accelerometer (x, y, z)
const float NORM_MEAN_ACCEL[]  = { 0.356837f, -0.049351f, 0.022777f };
const float NORM_SCALE_ACCEL[] = { 0.614898f,  0.556168f, 0.759103f };

// Gyroscope (x, y, z)
const float NORM_MEAN_GYRO[]   = { 0.792414f,  0.271208f, -0.231060f };
const float NORM_SCALE_GYRO[]  = { 57.045824f, 122.999524f, 98.049251f };

// Magnetometer (x, y, z)
const float NORM_MEAN_MAG[]    = { 7.071025f,  -19.211967f, -5.879854f };
const float NORM_SCALE_MAG[]   = { 23.375511f, 22.936115f, 25.879325f };

// -----------------------------------------------------------------------------
// BLE configuration
// -----------------------------------------------------------------------------

// Service UUID
BLEService postureService("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");

// RX characteristic: phone/tablet -> board (Write / Write Without Response)
BLECharacteristic rxCharacteristic(
  "6E400002-B5A3-F393-E0A9-E50E24DCCA9E",
  BLEWrite | BLEWriteWithoutResponse,
  20
);

// TX characteristic: board -> phone/tablet (Notify)
BLECharacteristic txCharacteristic(
  "6E400003-B5A3-F393-E0A9-E50E24DCCA9E",
  BLENotify,
  30
);

// -----------------------------------------------------------------------------
// TFLite Setting
// -----------------------------------------------------------------------------

tflite::ErrorReporter* error_reporter = nullptr;
tflite::AllOpsResolver resolver;             // Resolves all built-in ops
const tflite::Model* model            = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input             = nullptr;
TfLiteTensor* model_output            = nullptr;

// Tensor arena: 15 KB
constexpr int kTensorArenaSize = 15 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Model / data parameters
const int kWindowSize    = 100;   // 2 seconds @ 50 Hz
const int kInputChannels = 3;     // x, y, z
const int kNumClasses    = 5;

// classes
const char* CLASSES[] = {"sitting", "supine", "prone", "side", "unknown"};

// -----------------------------------------------------------------------------
// IMU sampling configuration
// -----------------------------------------------------------------------------

float sampleRate = 50.0f;              // 50 Hz
long interval    = 1000.0 / sampleRate;
long lastMillis  = 0;

// -----------------------------------------------------------------------------
// Quantization parameters (int8 model)
// -----------------------------------------------------------------------------

float input_scale       = 0.0f;
int   input_zero_point  = 0;
float output_scale      = 0.0f;
int   output_zero_point = 0;

// -----------------------------------------------------------------------------
// Forward declaration
// -----------------------------------------------------------------------------

void perform_inference(char sensor_type);

// -----------------------------------------------------------------------------
// BLE connection callbacks (optional, for logging)
// -----------------------------------------------------------------------------

void onBLEConnected(BLEDevice central) {
  Serial.print("BLE: Connected to central device: ");
  Serial.println(central.address());
}

void onBLEDisconnected(BLEDevice central) {
  Serial.print("BLE: Disconnected from central device: ");
  Serial.println(central.address());
  BLE.advertise();
  Serial.println("BLE: Advertising again, waiting for connection...");
}

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Posture Detector BLE Starting...");

  // 1) TFLite error reporter (writes errors to Serial)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // 2) Initialize IMU (BMI270 + BMM150)
  if (!IMU.begin()) {
    Serial.println("!!! Error: Failed to initialize IMU !!!");
    while (1);
  }

  // 3) Get TFLite model
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error: Model schema mismatch !");
    while (1);
  }

  // 4) Create a single static MicroInterpreter instance and keep a pointer
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 5) Allocate tensor buffers in the tensor arena
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Error: AllocateTensors() failed !");
    while (1);
  }

  // 6) Get input / output tensors and quantization parameters
  model_input  = interpreter->input(0);
  model_output = interpreter->output(0);

  input_scale       = model_input->params.scale;
  input_zero_point  = model_input->params.zero_point;
  output_scale      = model_output->params.scale;
  output_zero_point = model_output->params.zero_point;

  // 7) Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Error: Failed to initialize BLE !");
    while (1);
  }

  BLE.setEventHandler(BLEConnected,    onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);

  BLE.setLocalName("SW_PostureProject");
  BLE.setAdvertisedService(postureService);

  postureService.addCharacteristic(rxCharacteristic);
  postureService.addCharacteristic(txCharacteristic);
  BLE.addService(postureService);

  BLE.advertise();

  Serial.println("\n--- Arduino Ready (BLE) ---");
  Serial.println("Waiting for command (1=Accel, 2=Gyro, 3=Mag)...");
}

// -----------------------------------------------------------------------------
// Main loop
// -----------------------------------------------------------------------------

void loop() {
  // Service BLE stack regularly
  BLE.poll();

  // Check if RX characteristic was written by the central
  if (rxCharacteristic.written()) {
    int len = rxCharacteristic.valueLength();

    if (len > 0) {
      const uint8_t* data = rxCharacteristic.value();
      char cmd = (char)data[0];

      if (cmd == '1' || cmd == '2' || cmd == '3') {
        Serial.print("Received command via BLE: ");
        Serial.println(cmd);
        perform_inference(cmd);
      } else {
        Serial.println("Invalid command received via BLE (expected '1', '2', or '3').");
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Inference routine
// - sensor_type: '1' = accel, '2' = gyro, '3' = mag
// -----------------------------------------------------------------------------

void perform_inference(char sensor_type) {
  float ax, ay, az, gx, gy, gz, mx, my, mz;

  Serial.println("Collecting window for inference...");

  // Collect a 2 s window (100 samples 50 Hz)
  for (int i = 0; i < kWindowSize; i++) {
    // Enforce sampling interval while still servicing BLE
    while (millis() - lastMillis < interval) {
      BLE.poll();
    }
    lastMillis = millis();

    // Ensure all sensors have fresh data
    if (!IMU.accelerationAvailable() ||
        !IMU.gyroscopeAvailable()   ||
        !IMU.magneticFieldAvailable()) {
      i--;
      continue;
    }

    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    IMU.readMagneticField(mx, my, mz);

    float x_norm = 0.0f, y_norm = 0.0f, z_norm = 0.0f;

    // Select appropriate normalization constants based on sensor_type
    if (sensor_type == '1') {
      // Accelerometer
      x_norm = (ax - NORM_MEAN_ACCEL[0]) / NORM_SCALE_ACCEL[0];
      y_norm = (ay - NORM_MEAN_ACCEL[1]) / NORM_SCALE_ACCEL[1];
      z_norm = (az - NORM_MEAN_ACCEL[2]) / NORM_SCALE_ACCEL[2];
    } else if (sensor_type == '2') {
      // Gyroscope
      x_norm = (gx - NORM_MEAN_GYRO[0]) / NORM_SCALE_GYRO[0];
      y_norm = (gy - NORM_MEAN_GYRO[1]) / NORM_SCALE_GYRO[1];
      z_norm = (gz - NORM_MEAN_GYRO[2]) / NORM_SCALE_GYRO[2];
    } else {
      // Magnetometer ('3')
      x_norm = (mx - NORM_MEAN_MAG[0]) / NORM_SCALE_MAG[0];
      y_norm = (my - NORM_MEAN_MAG[1]) / NORM_SCALE_MAG[1];
      z_norm = (mz - NORM_MEAN_MAG[2]) / NORM_SCALE_MAG[2];
    }

    // Quantize normalized floats to int8 using input_scale and zero_point
    model_input->data.int8[i * kInputChannels + 0] =
      (int8_t)round((x_norm / input_scale) + input_zero_point);
    model_input->data.int8[i * kInputChannels + 1] =
      (int8_t)round((y_norm / input_scale) + input_zero_point);
    model_input->data.int8[i * kInputChannels + 2] =
      (int8_t)round((z_norm / input_scale) + input_zero_point);
  }

  Serial.println("Window collected, running inference...");

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Find class with maximum score (int8)
  int   predicted_class_index = -1;
  int8_t max_score_int8       = -128;

  for (int i = 0; i < kNumClasses; i++) {
    int8_t s = model_output->data.int8[i];
    if (s > max_score_int8) {
      max_score_int8       = s;
      predicted_class_index = i;
    }
  }

  // Prepare result
  char resultBuffer[30];
  sprintf(resultBuffer, "Prediction: %s", CLASSES[predicted_class_index]);

  // Send prediction back
  txCharacteristic.writeValue((uint8_t*)resultBuffer, strlen(resultBuffer));

  Serial.print("Sent via BLE: ");
  Serial.println(resultBuffer);
}
