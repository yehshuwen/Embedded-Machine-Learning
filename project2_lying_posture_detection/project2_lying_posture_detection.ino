// -----------------------------------------------------------------------------
// Project 2: Lying Posture Detection System
// Author: Shu-Wen yeh
// Date: October 31, 2025
//
// Description:
// This program uses the onboard IMU (BMI270) to detect a user's lying
// posture (supine, prone, or side) based on accelerometer data. It provides
// real-time visual feedback by blinking the built-in LED a specific number
// of times for each detected posture.
// -----------------------------------------------------------------------------

#include <Arduino_BMI270_BMM150.h>

// Define constants to present each posture
const int SUPINE_POSTURE = 1;
const int PRONE_POSTURE = 2;
const int SIDE_POSTURE = 3;
const int UNKOWN_POSTURE = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { };

  // Init the IMU sencor
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Set the built-in LED pin as an output
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("Real-time posture detection running...");
}

void loop() {
  // Declare acceleration values for each axis
  float ax, ay, az;

  // Check if acceleration data is available or not
  if (IMU.accelerationAvailable()) {
    // Read the acceleration data
    IMU.readAcceleration(ax, ay, az);

    // Print raw data for observation
    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.print(",");
    Serial.println(az);

    // Init the current posture as UNKOWN_POSTURE
    int currentPosture = UNKOWN_POSTURE;

    if (az > 0.8) {
      currentPosture = SUPINE_POSTURE;
      Serial.println("Posture: Supine");
    } else if (az < -0.8) {
      currentPosture = PRONE_POSTURE;
      Serial.println("Posture: Prone");
    } else if (abs(ax) > 0.8 || abs(ay) > 0.8 ) {
      currentPosture = SIDE_POSTURE;
      Serial.println("Posture: Side");
    } else {
      Serial.println(">>>> Posture: Unknown <<<<");
    }

    // LED logic
    switch (currentPosture) {
      case SUPINE_POSTURE:
        blinkLED(1);  // Blink once for supine
        break;
      case PRONE_POSTURE:
        blinkLED(2);  // Blink twice for prone
        break;
      case SIDE_POSTURE:
        blinkLED(3);  // Blink 3-times for side
        break;
      case UNKOWN_POSTURE:
      default:
        digitalWrite(LED_BUILTIN, LOW); // For any other case, LED is turn off
        break;
    }

    // Add a significant delay to control the update frequency
    delay(2000);
  }
}

// Function for controling LED blink times
void blinkLED(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN, HIGH);  // LED on
    delay(200);                       // Keep it on for 0.2 seconds
    digitalWrite(LED_BUILTIN, LOW);   // LED off
    delay(200);                       // Keep it off for 0.2 seconds
  }
}
