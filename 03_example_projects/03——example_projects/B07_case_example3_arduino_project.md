# B07 Case Example 3: Arduino LED Project

## 1. Project Overview
This is a beginner-level Arduino example project.
The goal of this circuit is to control an LED light, turn the LED on and off repeatedly.

## 2. Required Components
- Arduino Uno board ×1
- LED ×1
- 220Ω resistor ×1
- Breadboard ×1
- Jumper wires

## 3. Circuit Wiring Instruction
1. The anode(long leg) of LED connects to Arduino digital pin 13 through the resistor.
2. The cathode(short leg) of LED connects to GND.

## 4. Arduino Source Code
```arduino
// define LED pin
const int ledPin = 13;

void setup() {
  pinMode(ledPin, OUTPUT);
}

void loop() {
  digitalWrite(ledPin, HIGH);   // LED ON
  delay(1000);                  // wait 1 second
  digitalWrite(ledPin, LOW);    // LED OFF
  delay(1000);                  // wait 1 second
}