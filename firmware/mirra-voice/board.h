/*
 * Pin map of the Onju Voice V3 board (ESP32-S3 with PSRAM).
 * Taken from the open source onju-v2 project (MIT license, github.com/justLV/onju-v2).
 */
#pragma once

#define I2S_BCK_PIN 18
#define I2S_WS_PIN 13
#define I2S_IN_PIN 17   // INMP441 microphone
#define I2S_OUT_PIN 12  // MAX98357A amplifier
#define SPEAKER_EN_PIN 21
#define MUTE_PIN 38     // hardware mute switch
#define LED_PIN 11
#define LED_COUNT 6

#define TOUCH_LEFT T2
#define TOUCH_CENTER T3
#define TOUCH_RIGHT T4
