/*
 * Copyright (c) 2026 MirraNET, Niklas Linz. All rights reserved.
 * Licensed under the GNU Lesser General Public License v3.0 (LGPLv3).
 *
 * Streaming inference follows the micro_wake_word component of ESPHome (Apache-2.0 / MIT), which defines
 * how microWakeWord models are run.
 */
#include "wakeword.h"

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_allocator.h>
#include <tensorflow/lite/micro/micro_resource_variable.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <esp_heap_caps.h>

#include "src/microfrontend/frontend.h"
#include "src/microfrontend/frontend_util.h"

namespace {
constexpr size_t FEATURE_SIZE = 40;
constexpr size_t SAMPLES_PER_STEP = 160;  // 10 ms
constexpr size_t TENSOR_ARENA_SIZE = 64 * 1024;
constexpr size_t VARIABLE_ARENA_SIZE = 1024;
constexpr int16_t SLICES_BEFORE_DETECTION = 100;  // one second

FrontendConfig frontendConfig;
FrontendState frontendState;
tflite::MicroMutableOpResolver<20> resolver;
tflite::MicroInterpreter *interpreter = nullptr;
uint8_t *tensorArena = nullptr;
uint8_t *variableArena = nullptr;

bool registerOps() {
  return resolver.AddCallOnce() == kTfLiteOk && resolver.AddVarHandle() == kTfLiteOk &&
         resolver.AddReshape() == kTfLiteOk && resolver.AddReadVariable() == kTfLiteOk &&
         resolver.AddStridedSlice() == kTfLiteOk && resolver.AddConcatenation() == kTfLiteOk &&
         resolver.AddAssignVariable() == kTfLiteOk && resolver.AddConv2D() == kTfLiteOk &&
         resolver.AddMul() == kTfLiteOk && resolver.AddAdd() == kTfLiteOk && resolver.AddMean() == kTfLiteOk &&
         resolver.AddFullyConnected() == kTfLiteOk && resolver.AddLogistic() == kTfLiteOk &&
         resolver.AddQuantize() == kTfLiteOk && resolver.AddDepthwiseConv2D() == kTfLiteOk &&
         resolver.AddAveragePool2D() == kTfLiteOk && resolver.AddMaxPool2D() == kTfLiteOk &&
         resolver.AddPad() == kTfLiteOk && resolver.AddPack() == kTfLiteOk && resolver.AddSplitV() == kTfLiteOk;
}
}  // namespace

bool WakeWord::begin(const uint8_t *model, float cutoff, uint8_t window) {
  cutoff_ = (uint8_t)constrain((int)(cutoff * 255.0f + 0.5f), 1, 255);
  window_ = constrain(window, (uint8_t)1, (uint8_t)sizeof(probabilities_));

  frontendConfig.window.size_ms = 30;
  frontendConfig.window.step_size_ms = 10;
  frontendConfig.filterbank.num_channels = FEATURE_SIZE;
  frontendConfig.filterbank.lower_band_limit = 125.0f;
  frontendConfig.filterbank.upper_band_limit = 7500.0f;
  frontendConfig.noise_reduction.smoothing_bits = 10;
  frontendConfig.noise_reduction.even_smoothing = 0.025f;
  frontendConfig.noise_reduction.odd_smoothing = 0.06f;
  frontendConfig.noise_reduction.min_signal_remaining = 0.05f;
  frontendConfig.pcan_gain_control.enable_pcan = 1;
  frontendConfig.pcan_gain_control.strength = 0.95f;
  frontendConfig.pcan_gain_control.offset = 80.0f;
  frontendConfig.pcan_gain_control.gain_bits = 21;
  frontendConfig.log_scale.enable_log = 1;
  frontendConfig.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&frontendConfig, &frontendState, 16000)) {
    Serial.println("[wake] frontend init failed");
    return false;
  }

  const tflite::Model *tfModel = tflite::GetModel(model);
  if (tfModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[wake] model schema not supported");
    return false;
  }
  if (!registerOps()) {
    Serial.println("[wake] op registration failed");
    return false;
  }
  /* internal RAM is much faster than PSRAM for the arena */
  tensorArena = (uint8_t *)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!tensorArena) tensorArena = (uint8_t *)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
  variableArena = (uint8_t *)heap_caps_aligned_alloc(16, VARIABLE_ARENA_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!tensorArena || !variableArena) {
    Serial.println("[wake] out of memory");
    return false;
  }
  tflite::MicroAllocator *variableAllocator = tflite::MicroAllocator::Create(variableArena, VARIABLE_ARENA_SIZE);
  tflite::MicroResourceVariables *variables = tflite::MicroResourceVariables::Create(variableAllocator, 20);
  interpreter = new tflite::MicroInterpreter(tfModel, resolver, tensorArena, TENSOR_ARENA_SIZE, variables);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[wake] tensor allocation failed");
    return false;
  }
  TfLiteTensor *input = interpreter->input(0);
  TfLiteTensor *output = interpreter->output(0);
  if (input->dims->size != 3 || input->dims->data[0] != 1 || input->dims->data[2] != (int)FEATURE_SIZE ||
      input->type != kTfLiteInt8 || output->type != kTfLiteUInt8) {
    Serial.println("[wake] model has unexpected input/output");
    return false;
  }
  Serial.printf("[wake] model ready, stride %d, arena %u bytes used, cutoff %.2f, window %u\n", input->dims->data[1],
                (unsigned)interpreter->arena_used_bytes(), cutoff, window_);
  ready_ = true;
  reset();
  return true;
}

void WakeWord::reset() {
  memset(probabilities_, 0, sizeof(probabilities_));
  ignoreSlices_ = -SLICES_BEFORE_DETECTION;
  lastProbability_ = 0;
}

bool WakeWord::inferFeatures(const int8_t *features) {
  TfLiteTensor *input = interpreter->input(0);
  uint8_t stride = input->dims->data[1];
  strideStep_ %= stride;
  memcpy(tflite::GetTensorData<int8_t>(input) + FEATURE_SIZE * strideStep_, features, FEATURE_SIZE);
  strideStep_++;
  if (strideStep_ >= stride) {
    if (interpreter->Invoke() != kTfLiteOk) return false;
    probabilityIndex_ = (probabilityIndex_ + 1) % window_;
    lastProbability_ = interpreter->output(0)->data.uint8[0];
    probabilities_[probabilityIndex_] = lastProbability_;
  }
  if (probabilities_[probabilityIndex_] < cutoff_) {
    /* cool-off: only count slices below the cutoff, so one utterance is not detected twice */
    ignoreSlices_ = min((int16_t)(ignoreSlices_ + 1), (int16_t)0);
  }
  if (ignoreSlices_ < 0) return false;
  uint32_t sum = 0;
  for (uint8_t i = 0; i < window_; i++) sum += probabilities_[i];
  if (sum > (uint32_t)cutoff_ * window_) {
    reset();
    return true;
  }
  return false;
}

bool WakeWord::process(const int16_t *samples, size_t count) {
  if (!ready_) return false;
  bool detected = false;
  while (count > 0) {
    size_t take = min(count, SAMPLES_PER_STEP - pendingCount_);
    memcpy(pending_ + pendingCount_, samples, take * sizeof(int16_t));
    pendingCount_ += take;
    samples += take;
    count -= take;
    if (pendingCount_ < SAMPLES_PER_STEP) break;

    size_t read = 0;
    FrontendOutput out = FrontendProcessSamples(&frontendState, pending_, SAMPLES_PER_STEP, &read);
    /* keep samples the frontend did not consume (it always consumes the full step here) */
    if (read < SAMPLES_PER_STEP) memmove(pending_, pending_ + read, (SAMPLES_PER_STEP - read) * sizeof(int16_t));
    pendingCount_ = SAMPLES_PER_STEP - read;
    if (out.size != FEATURE_SIZE) continue;

    int8_t features[FEATURE_SIZE];
    for (size_t i = 0; i < FEATURE_SIZE; i++) {
      /* input = feature * 256 / (25.6 * 26) - 128, see microWakeWord / ESPHome */
      int32_t value = ((int32_t)out.values[i] * 256 + 333) / 666 - 128;
      features[i] = (int8_t)constrain(value, -128, 127);
    }
    if (inferFeatures(features)) detected = true;
  }
  return detected;
}
