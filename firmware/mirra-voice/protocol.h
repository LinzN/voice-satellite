/*
 * Wire protocol between the satellite and the STEM voice-satellite plugin.
 *
 * Every frame: 1 byte type, 4 bytes payload length (big endian), payload.
 * Audio is PCM 16 bit little endian, mono, 16000 Hz in both directions.
 * Control payloads are small JSON objects.
 */
#pragma once

#define PROTOCOL_VERSION 1
#define MAX_FRAME_PAYLOAD 65536

/* device -> server */
#define F_HELLO 0x01  // {"id","fw","name","mac","proto","hmac"}  hmac = HMAC-SHA256(key, nonce) as hex
#define F_AUDIO 0x02  // microphone PCM while listening
#define F_EVENT 0x03  // {"event":"listen_start"|"listen_stop"|"cancel"|"playback_done"|"mute","value":..}
#define F_PING 0x04

/* server -> device */
#define F_CHALLENGE 0x80  // {"nonce","proto"}
#define F_WELCOME 0x81    // {"name","state"}
#define F_PLAY 0x82       // speaker PCM
#define F_PLAY_END 0x83   // {"listen":bool,"cancelled":bool}
#define F_STATE 0x84      // {"state":"idle"|"thinking"|"pending"|"denied"|"error"}
#define F_STOP_LISTEN 0x85
#define F_PONG 0x86
#define F_CMD 0x87        // {"cmd":"volume"|"chime"|"reboot","value":n}
#define F_ERROR 0x88      // {"message"}
