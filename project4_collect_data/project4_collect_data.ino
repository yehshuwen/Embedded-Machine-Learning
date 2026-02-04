// -----------------------------------------------------------------------------
// Project 4: Posture Detection System
// Author: Shu-Wen yeh
// Date: Nov 14, 2025
//
// Description:
// This program uses the onboard IMU BMI270(Acceleration, Gyroscope) to detect a user's
// posture (supine, prone, side, sit, or unknown) based on sensor data.
// Collect these data from Arduino board and use machine learning to build a
// model to detect the posture.
// -----------------------------------------------------------------------------

#include <Arduino_BMI270_BMM150.h>

// fixed sample rate
float sampleRate = 50.0;
long interval = 1000.0 / sampleRate; // 20 milliseconds
long lastMillis = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while(1);
  }

  Serial.println("ax, ay, az, gx, gy, gz, mx, my, mz");
}

void loop() {
  float ax, ay, az ,gx, gy, gz, mx, my, mz;
  if (millis() - lastMillis >= interval) {
    // reset timer
    lastMillis = millis();
    if (IMU.accelerationAvailable() && 
        IMU.gyroscopeAvailable() && 
        IMU.magneticFieldAvailable()) {
          
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      IMU.readMagneticField(mx, my, mz);

      Serial.print(ax, 4);
      Serial.print(",");
      Serial.print(ay, 4);
      Serial.print(",");
      Serial.print(az, 4);
      Serial.print(",");
      Serial.print(gx, 4);
      Serial.print(",");
      Serial.print(gy, 4);
      Serial.print(",");
      Serial.print(gz, 4);
      Serial.print(",");
      Serial.print(mx, 4);
      Serial.print(",");
      Serial.print(my, 4);
      Serial.print(",");
      Serial.println(mz, 4);
    }
  }
}
