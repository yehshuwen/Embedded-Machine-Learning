/**
 * @file project1_ble_control.ino
 * @author Shu-Wen Yeh
 * @brief Implements a state machine on an Arduino Nano 33 BLE Sense,
 * controlled wirelessly via a Bluetooth Low Energy (BLE) connection.
 * @date October 24, 2025
 * * @description
 * This program models a finite state machine with four states: DARK, RED, BLUE, and GREEN.
 * The state is visually represented by the onboard RGB LED.
 * Transitions are triggered by two types of events:
 * 1. A software "switch" press, simulated by writing a character ('s') to a BLE characteristic.
 * 2. Automatic timeouts, managed by a non-blocking timer using the millis() function.
 * * A mobile application like nRF Connect can be used to connect to the Arduino and
 * write to the characteristic, thereby controlling the state machine.
 */

#include <ArduinoBLE.h>

// Create a BLE service
BLEService stateMachineService("dd51e079-41e4-3dbf-7883-2afe68e7ed6f"); 

// Create a switch characteristic to recive the message
BLECharCharacteristic switchCharacteristic("dd51e079-41e4-3dbf-7883-2afe68e7ed6f", BLEWrite);

// Define RGB LED Pins
const int RED_PIN = 22;
const int GREEN_PIN = 23;
const int BLUE_PIN = 24;

// Use enum to express state
enum State {
  DARK_STATE,
  RED_STATE,
  BLUE_STATE,
  GREEN_STATE
};

State currentState = DARK_STATE;
unsigned long stateEnterTime = 0;

void setLedColor(int r, int g, int b);

void setup() {
  Serial.begin(115200);
  while (!Serial) {};

  // Init BLE library
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  // Set BLE Name
  BLE.setLocalName("SWTrigger");
  // Set advertised service
  BLE.setAdvertisedService(stateMachineService);

  // Add characteristic to service
  stateMachineService.addCharacteristic(switchCharacteristic);
  // Add service to device
  BLE.addService(stateMachineService);

  // Set an init value for cahracteristic
  switchCharacteristic.writeValue(' ');

  // Start advertising
  BLE.advertise();
  Serial.println("Bluetooth device is advertising, waiting for connections...");

  // Set LED pins to output mode
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);

  // Init LED color as dark
  setLedColor(0, 0, 0);
  currentState = DARK_STATE;
}

void loop() {
  // Wait for a device to connect
  BLEDevice central = BLE.central();

  // Set a boolean switch pressed flag
  bool switchPressed = false;
  
  // Check if device is connected
  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    // This while loop will run if device is connected
    while (central.connected()) {
      // Check if switch characteristic has written
      if (switchCharacteristic.written()) {
        // Read the wirtten value
        char value = switchCharacteristic.value();

        // If value is 's' then set switch pressed flag to 'true'
        if (value == 's') {
          switchPressed = true;
          Serial.println("Switch triggered via BLE!");
        }
      }

      // This section handle RGB LED logic if switch is pressed
      if (switchPressed) {
          switch (currentState) {
              case DARK_STATE:
                  Serial.println("Transition: DARK -> RED");
                  currentState = RED_STATE;
                  stateEnterTime = millis(); // Record the start time 
                  setLedColor(255, 0, 0); // Set LED color to red
                  break;
              case RED_STATE:
                  Serial.println("Transition: RED -> BLUE");
                  currentState = BLUE_STATE;
                  stateEnterTime = millis(); // Record the start time
                  setLedColor(0, 0, 255); // Set LED color to blue
                  break;
              case BLUE_STATE:
                  Serial.println("Transition: BLUE -> GREEN");
                  currentState = GREEN_STATE;
                  stateEnterTime = millis(); // Record the start time
                  setLedColor(0, 255, 0); // Set LED color to green
                  break;
              case GREEN_STATE:
                  Serial.println("Transition: GREEN -> DARK");
                  currentState = DARK_STATE; // Record the start time
                  setLedColor(0, 0, 0); // Set LED color to dark
                  break;
          }
          switchPressed = false; // Reset switch pressed flag
      } else { // when switch is not prerssed this section will check if it timeouts
          switch (currentState) {
              case RED_STATE:
                  if (millis() - stateEnterTime >= 5000) { // 5 seconds
                      Serial.println("Transition: RED -> DARK (Timeout)");
                      currentState = DARK_STATE;
                      setLedColor(0, 0, 0);
                  }
                  break;
              case BLUE_STATE:
                  if (millis() - stateEnterTime >= 4000) { // 4 seconds
                      Serial.println("Transition: BLUE -> RED (Timeout)");
                      currentState = RED_STATE;
                      stateEnterTime = millis();
                      setLedColor(255, 0, 0);
                  }
                  break;
              case GREEN_STATE:
                  if (millis() - stateEnterTime >= 3000) { // 3 seconds
                      Serial.println("Transition: GREEN -> BLUE (Timeout)");
                      currentState = BLUE_STATE;
                      stateEnterTime = millis();
                      setLedColor(0, 0, 255);
                  }
                  break;
          }
      }
      delay(10); // A small delay time is added to maintain the connection
    }
    
    // This code runs if the central device disconnects.
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}

// Function to control the LED color
void setLedColor(int r, int g, int b) {
  analogWrite(RED_PIN, 255 - r);
  analogWrite(GREEN_PIN, 255 - g);
  analogWrite(BLUE_PIN, 255 - b);
}