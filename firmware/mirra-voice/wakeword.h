/*
 * Copyright (c) 2026 MirraNET, Niklas Linz. All rights reserved.
 * Licensed under the GNU Lesser General Public License v3.0 (LGPLv3).
 *
 * On-device wake word detection with a microWakeWord streaming model (TFLite Micro).
 * The feature pipeline (TFLite micro frontend, 30 ms windows every 10 ms, 40 channels) and the
 * input quantization match the training of microWakeWord models.
 */
#pragma once

#include <Arduino.h>

class WakeWord {
 public:
  /*
   * @param model     the .tflite model (must stay valid)
   * @param cutoff    detection threshold 0..1 of the averaged probability
   * @param window    number of recent model outputs that are averaged
   */
  bool begin(const uint8_t *model, float cutoff, uint8_t window);

  /* Feed 16 kHz mono samples. Returns true once when the wake word was detected. */
  bool process(const int16_t *samples, size_t count);

  /* Forget recent audio, e.g. after the speaker played something. Detection needs one second of new audio. */
  void reset();

  bool ready() const { return ready_; }
  uint8_t lastProbability() const { return lastProbability_; }
  /* highest probability since the last call (for tuning the cutoff) */
  uint8_t takeMaxProbability() { uint8_t value = maxProbability_; maxProbability_ = 0; return value; }

 private:
  bool inferFeatures(const int8_t *features);

  bool ready_ = false;
  uint8_t cutoff_ = 230;
  uint8_t window_ = 5;
  uint8_t probabilities_[16] = {0};
  uint8_t probabilityIndex_ = 0;
  int16_t ignoreSlices_ = -100;
  uint8_t strideStep_ = 0;
  uint8_t lastProbability_ = 0;
  uint8_t maxProbability_ = 0;
  int16_t pending_[160];
  size_t pendingCount_ = 0;
};
