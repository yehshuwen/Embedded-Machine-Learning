// -----------------------------------------------------------------------------
// Project 3: Posture Detection System
// Author: Shu-Wen yeh
// Date: Nov 07, 2025
//
// Description:
// This program uses the onboard IMU BMI270(Acceleration, Gyroscope) to detect a user's
// posture (supine, prone, side, sit, or unknown) based on sensor data.
// Collect these data from Arduino board and use machine learning to build a
// model to detect the posture.
// -----------------------------------------------------------------------------

#include <Arduino_BMI270_BMM150.h>

// fixed sample rate
float sampleRate = 100.0;
long interval = 1000.0 / sampleRate; // 10 milliseconds
long lastMillis = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while(1);
  }

  Serial.print("IMU initialized. Manually sampling at ");
  Serial.print(sampleRate);
  Serial.println(" Hz.");

  Serial.println("IMU initialized. Send any character to start logging...");
}

void loop() {
  if (Serial.available()>0) {
    Serial.read();
  }

  float ax, ay, az, gx, gy, gz;

  if (millis() - lastMillis >= interval) {
    // reset timer
    lastMillis = millis();

      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);

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
      Serial.println(gz, 4);
    }
  }
}
