# voice-satellite

Talk to **Mirra** through the smart speakers in the house. The firmware turns an
[Onju Voice V3](https://github.com/justLV/onju-v2) board inside a Google Home Mini shell into a voice satellite. The
STEM plugin in this repository handles speech detection, transcription, the Mirra turn and text to speech.

The speaker hardware keeps working: speaker, microphone, the six LEDs, the three touch areas and the mute switch.

```
 Home Mini (Onju V3)                     STEM                                  OpenAI
 ┌──────────────────┐   TCP 3100   ┌────────────────────┐   MirraVoice   ┌───────────────────┐
 │ tap → mic 16 kHz ├─────────────►│ voice-satellite    ├──────────────► │ transcription     │
 │ LEDs, touch, mute│              │  end of speech     │                │ Mirra agent+tools │
 │ speaker ◄────────┤◄─────────────┤  24→16 kHz, stream │◄───────────────┤ text to speech     │
 └──────────────────┘   PCM 16 kHz └────────────────────┘                └───────────────────┘
```

## Using the speaker

| Action | What happens |
|---|---|
| Say **"Hey Mirra"** | Mirra listens right away (no beep, just keep talking: "Hey Mirra, mach das Licht im Flur an") |
| Tap the center | Short beep, Mirra listens. Stop talking and she answers on her own (end of speech is detected) |
| Tap the center while listening | Stops listening right away |
| Tap the center while Mirra speaks | Interrupts her and listens again |
| Tap the center while Mirra thinks | Cancels the question |
| Tap left / right | Volume down / up (stored on the device) |
| Mute switch | Microphone off (also the wake word); the speaker still plays announcements |

If Mirra's answer ends with a question, the speaker listens again by itself. While the speaker plays, it does not
react to the wake word (no echo cancellation on the board), tap to interrupt.

| LEDs | Meaning |
|---|---|
| off | ready |
| cyan, following your voice | listening |
| purple, rotating | Mirra thinks |
| white, following the voice | Mirra speaks |
| orange, slow pulse on all LEDs | device waits for the owner's approval |
| one dim orange LED pulsing | no connection to STEM (Wi-Fi or server) |
| one red LED | denied by the owner, or (last LED) microphone muted |
| white bar | volume after a tap left/right |
| short red blink | error, e.g. tap while muted or the request failed |
| blue | firmware update over the air |

## Installation

### 1. Plugin

Requires STEM with the **mirra** plugin (version with `MirraVoice`).

1. Build with `mvn package` and put `target/voice-satellite.jar` into the plugin folder of STEM.
2. Start STEM once. The plugin creates its config and a random device key, and logs it:
   `New device key generated (server.deviceKey ...): 3f9c...`
3. Open port **3100/TCP** on the STEM host for the local network only (not to the internet).

`config.yml`:

```yaml
server:
  bindAddress: 0.0.0.0
  port: 3100
  deviceKey: 3f9c...            # shared secret, the same value goes into config.h of every device
  onlyPrivateNetworks: true     # refuse connections from public addresses
  maxDevices: 16
speech:
  endSilenceMs: 900             # pause that ends a question
  noSpeechTimeoutMs: 6000       # nobody speaks after the tap: stop listening
  maxListenMs: 15000            # longest question
  speechFactor: 3.0             # how much louder than the background speech must be
  minSpeechLevel: 400           # absolute minimum level for speech (16 bit RMS)
conversation:
  followUpOnQuestion: true      # listen again if Mirra asks something
  speechChunkChars: 300         # answers are spoken sentence by sentence in chunks of this size
devices:
  mirra-voice-a1b2c3:
    name: Küche                 # room name, written on first connect, can be changed here
```

Voice and transcription models come from Mirra's config (`voice.*`, `openAI.transcriptionModel`).

### 2. Firmware

The sketch is in [`firmware/mirra-voice`](firmware/mirra-voice).

1. Copy `config.example.h` to `config.h` and set Wi-Fi, `SERVER_HOST` (fixed IP of STEM), `DEVICE_KEY`
   (the `server.deviceKey` from above), optionally `DEVICE_NAME` and `OTA_PASSWORD`.
2. Optional: copy `hey_mirra_model.h` (from the wake word training, see below) next to `mirra-voice.ino`.
   Without it the firmware works with tapping only.
3. Arduino IDE: install the **esp32** board package by Espressif (3.x) and the library **Adafruit NeoPixel**.
   Board **ESP32S3 Dev Module**, **USB CDC On Boot: Enabled**, **PSRAM: OPI PSRAM**,
   **Partition Scheme: Minimal SPIFFS (1.9MB APP with OTA)** (the wake word does not fit the default scheme).
4. Connect the board by USB (hold BOOT while plugging in if it does not show up) and upload.

With arduino-cli:

```sh
arduino-cli core install esp32:esp32
arduino-cli lib install "Adafruit NeoPixel"
arduino-cli compile --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,PSRAM=opi,PartitionScheme=min_spiffs firmware/mirra-voice
arduino-cli upload  --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,PSRAM=opi,PartitionScheme=min_spiffs -p /dev/ttyACM0 firmware/mirra-voice
```

The partition scheme can only be changed over USB; later updates work over Wi-Fi (Arduino OTA, host name `mirra-voice-xxxxxx`, password `OTA_PASSWORD`).

**Windows, one command:** `firmware/flash.ps1` (or double-click `firmware/flash.cmd`) downloads arduino-cli, the
ESP32 core and the library into `firmware/tools`, asks once for Wi-Fi, server, device key and OTA password
(stored in `firmware/settings.json`), writes `config.h`, takes the wake word model from `../wakeword` or
`../mirra-wakeword` if present, builds and flashes the board it finds on USB:

```powershell
.\flash.ps1 -Name "Küche"                     # USB
.\flash.ps1 -Name "Küche" -Ota 192.168.1.50   # later updates over Wi-Fi
.\flash.ps1 -Name "Küche" -Monitor            # and open the serial monitor
```

Settings in `config.h` you may need to adjust:

- `TOUCH_SENSITIVITY` if taps are missed (lower) or trigger by themselves (higher). The pads calibrate at
  start, so do not touch the speaker while it boots.
- `WAKE_WORD_CUTOFF` (default `0.90`): raise it if the speaker wakes up by itself, lower it if it does not react
  to "Hey Mirra". `WAKE_WORD_ENABLED 0` turns the wake word off.
- `MUTE_ACTIVE_LEVEL` if the mute switch works the wrong way round (`0`), or set `-1` to ignore it.

The serial monitor (115200 baud) shows the device id, Wi-Fi and connection state.

### 3. Wake word "Hey Mirra"

The wake word runs on the speaker itself: a [microWakeWord](https://github.com/kahrendt/microWakeWord) model
(TFLite Micro, about 60 KB) listens to the microphone while the speaker is idle. Nothing leaves the device until
the wake word is detected.

The model is trained on a PC with an NVIDIA GPU with the scripts in [`wakeword/`](wakeword): synthetic
"Hey Mirra" samples from German and English voices, look-alike phrases that must not wake it ("Mirra" alone,
"Hey Mia", "Hey Kira", "Hey Siri", ...), background noise and speech. In WSL2:

```sh
bash wakeword/train.sh
```

It takes a few hours and writes `hey_mirra_model.h` (copy it to `firmware/mirra-voice/`) and the test results
(`results/tflite_streaming_roc.txt`: false wake ups per hour and missed wake words for each cutoff, use them to
choose `WAKE_WORD_CUTOFF`). See [`wakeword/README.md`](wakeword/README.md).

The model is not in this repository: the background datasets it is trained with are for personal,
non-commercial use only.

### 4. Mirra

The first time somebody speaks on a new device, the owner gets Mirra's usual access request
(*"🔔 Neue Kontaktanfrage an Mirra – Sprachgerät: Küche (mirra-voice-a1b2c3) – Erste Nachricht: …"*); the speaker
says the pending message and pulses orange until it is approved ("ja, das darf").

**Everybody next to the speaker talks with the permissions of the profile the device belongs to.** Do not link a
speaker to your own profile. Give the speakers a profile of their own with what a shared device should do:

> Mirra, leg ein Profil "Haus" an, ordne ihm das Sprachgerät mirra-voice-a1b2c3 zu und gib ihm ASSISTANT,
> REMINDERS und HOME_LIGHTS.

Speaking on the devices from chats and tasks:

> Sag in der Küche, dass das Essen fertig ist.
> Jeden Morgen um 7 Uhr: sag im Schlafzimmer das Wetter an.

## Console command

```
voice list                          devices with state, firmware and address
voice say <device|all> <text>       speak a text
voice volume <device> <0-10>        set the volume
voice reboot <device>               restart a device
voice record <device|all> on|off    save every recording as WAV and do NOT send it to Mirra (beep instead),
                                    to record "Hey Mirra" with the speaker's own microphone for the training
voice capture <device|all> on|off   save every recording as WAV and answer normally, to check the microphone
```

Recordings are saved in `plugins/voice-satellite/recordings/<device>/` of STEM; the log shows their level
(peak, average, clipped samples). For the wake word training, tap, wait for the beep and say "Hey Mirra",
50-100 times from different places in the room; then copy the WAV files into `recordings/` of the training
folder.

## Protocol

TCP, every frame is `type (1 byte) | length (4 bytes, big endian) | payload`. Audio is PCM 16 bit little endian,
mono, 16 kHz in both directions. Control payloads are JSON.

| Type | Direction | Payload |
|---|---|---|
| `0x80` CHALLENGE | server → device | `{"nonce","proto"}` |
| `0x01` HELLO | device → server | `{"id","fw","name","mac","proto","hmac"}`, `hmac = HMAC-SHA256(deviceKey, nonce + ":" + id)` |
| `0x81` WELCOME | server → device | `{"name","state"}` state: `idle`, `pending`, `denied` |
| `0x02` AUDIO | device → server | microphone audio while listening |
| `0x03` EVENT | device → server | `listen_start`, `listen_stop`, `cancel`, `playback_done`, `mute` |
| `0x85` STOP_LISTEN | server → device | end of speech detected |
| `0x84` STATE | server → device | `idle`, `thinking`, `pending`, `denied`, `error` |
| `0x82` PLAY | server → device | speaker audio |
| `0x83` PLAY_END | server → device | `{"listen":bool}` end of the answer, or `{"cancelled":true}` as reply to `cancel` |
| `0x87` CMD | server → device | `{"cmd":"volume","value":n}`, `chime`, `reboot` |
| `0x04` / `0x86` | both | ping / pong every 10 s |
| `0x88` ERROR | server → device | `{"message"}` before closing |

The speaker buffers up to 90 s of answer in PSRAM; the server sends as fast as TCP allows. While the speaker
plays, the microphone is off (the board has no echo cancellation), so interrupting works by tapping.

## Security

- Devices prove the shared key with an HMAC challenge; the key itself is never sent. Audio is **not encrypted**
  on the LAN. Keep port 3100 internal and if possible put the speakers into a separate Wi-Fi/VLAN.
- A device with a valid key is still only a new guest in Mirra until the owner approves it.
- Anyone with physical access to a speaker can read the key from its flash. If one goes missing, change
  `server.deviceKey` and re-flash the others.
- See also the security documentation of Mirra (`docs/security/SECURITY.md`, risks R19–R21).

## Credits

Pin map and audio settings of the Onju V3 board from [onju-v2](https://github.com/justLV/onju-v2) by Justin
Alvey (MIT license). The firmware here is a separate implementation for STEM and Mirra.

## License

LGPLv3, see the headers of the source files.
