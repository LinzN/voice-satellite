/*
 * Copy this file to config.h and fill in your values. config.h is not committed.
 */
#pragma once

/* Wi-Fi */
#define WIFI_SSID "my-wifi"
#define WIFI_PASSWORD "my-password"

/* The STEM server running the voice-satellite plugin (fixed address) */
#define SERVER_HOST "192.168.1.10"
#define SERVER_PORT 3100

/* Shared secret, must match "deviceKey" in the plugin config. It is never sent over the network. */
#define DEVICE_KEY "change-me-to-a-long-random-string"

/* Optional room name shown to Mirra and the owner, e.g. "Küche". Empty: the server decides. */
#define DEVICE_NAME ""

/* Password for over-the-air updates from the Arduino IDE / arduino-cli. Empty disables OTA. */
#define OTA_PASSWORD "change-me"

/* Speaker volume 0..10 after the first boot; afterwards the left/right touch pads change it */
#define DEFAULT_VOLUME 6

/* Touch sensitivity: relative change of the touch reading that counts as a touch (0.05 .. 0.5) */
#define TOUCH_SENSITIVITY 0.15f

/* Mute switch: 1 if the switch pulls the pin HIGH when muted (Onju V3 default), 0 for LOW, -1 to ignore it */
#define MUTE_ACTIVE_LEVEL 1

/* LED brightness 0..255 */
#define LED_BRIGHTNESS 160

/* Wake word "Hey Mirra" (needs hey_mirra_model.h next to the sketch, see "Wake word" in the README).
 * Tapping the center pad works in any case. */
#define WAKE_WORD_ENABLED 1
/* higher = fewer false wake ups, but you may have to speak more clearly (0.5 .. 0.99) */
#define WAKE_WORD_CUTOFF 0.90f
/* number of model outputs (30 ms each) that are averaged */
#define WAKE_WORD_WINDOW 5
