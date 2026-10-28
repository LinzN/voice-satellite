/*
 * Copyright (c) 2026 MirraNET, Niklas Linz. All rights reserved.
 *
 * This file is part of the MirraNET project and is licensed under the
 * GNU Lesser General Public License v3.0 (LGPLv3).
 *
 * You may use, distribute and modify this code under the terms
 * of the LGPLv3 license. You should have received a copy of the
 * license along with this file. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>
 * or contact: niklas.linz@mirranet.de
 */

package de.linzn.voicesatellite;

import de.linzn.mirra.voice.MirraVoice;
import org.json.JSONObject;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Pattern;

/**
 * One connected satellite: authentication, recording with end of speech detection, the turn with Mirra
 * and streaming of the spoken answer.
 */
public final class SatelliteConnection implements Runnable {

    private static final int SOCKET_TIMEOUT_MS = 35_000;
    private static final int PLAY_FRAME_BYTES = 4096;
    private static final long MAX_RECORDING_BYTES = 30L * Protocol.SAMPLE_RATE * 2;
    private static final Pattern SENTENCE_END = Pattern.compile("(?<=[.!?…])\\s+");
    private static final Pattern NOT_SPOKEN = Pattern.compile("[\\p{So}\\p{Cs}\\p{Cn}*_#`~>|]+");

    private enum Mode { IDLE, LISTENING, THINKING, SPEAKING }

    private final VoiceSatellitePlugin plugin;
    private final SatelliteConfig config;
    private final Socket socket;
    private final ExecutorService workers;
    private final Object writeLock = new Object();
    private final ReentrantLock speechLock = new ReentrantLock(true);
    /* every new turn or cancel increments the generation; work of an older generation is dropped */
    private final AtomicInteger generation = new AtomicInteger();

    private OutputStream out;
    private volatile String deviceId;
    private volatile String deviceName;
    private volatile String firmware = "";
    private volatile Mode mode = Mode.IDLE;
    private volatile boolean muted;
    private volatile String accessState = "idle";
    private volatile boolean closed;

    /* set while a turn of this connection runs, so texts Mirra sends to the device during the turn
     * (e.g. the pending approval message) are spoken right away instead of waiting for the turn to end */
    private static final ThreadLocal<SatelliteConnection> TURN = new ThreadLocal<>();

    private SpeechDetector detector;
    private ByteArrayOutputStream recording;

    SatelliteConnection(VoiceSatellitePlugin plugin, SatelliteConfig config, Socket socket, ExecutorService workers) {
        this.plugin = plugin;
        this.config = config;
        this.socket = socket;
        this.workers = workers;
    }

    public String deviceId() {
        return this.deviceId;
    }

    public String deviceName() {
        return this.deviceName;
    }

    public String firmware() {
        return this.firmware;
    }

    public String remoteAddress() {
        return this.socket.getInetAddress().getHostAddress();
    }

    public boolean isOpen() {
        return !this.closed && !this.socket.isClosed();
    }

    @Override
    public void run() {
        try {
            this.socket.setSoTimeout(SOCKET_TIMEOUT_MS);
            this.socket.setTcpNoDelay(true);
            this.out = this.socket.getOutputStream();
            DataInputStream in = new DataInputStream(new BufferedInputStream(this.socket.getInputStream()));
            if (!this.authenticate(in)) {
                return;
            }
            while (!this.closed) {
                int type = in.readUnsignedByte();
                int length = in.readInt();
                if (length < 0 || length > Protocol.MAX_FRAME_PAYLOAD) {
                    throw new IOException("Frame too large: " + length);
                }
                byte[] payload = new byte[length];
                in.readFully(payload);
                this.handle(type, payload);
            }
        } catch (SocketTimeoutException e) {
            this.plugin.getPluginLogger().info("Voice device " + this.label() + " timed out");
        } catch (EOFException e) {
            this.plugin.getPluginLogger().info("Voice device " + this.label() + " disconnected");
        } catch (IOException e) {
            if (!this.closed) {
                this.plugin.getPluginLogger().info("Voice device " + this.label() + " lost: " + e.getMessage());
            }
        } catch (Exception e) {
            this.plugin.getPluginLogger().error("Voice device " + this.label() + " failed", e);
        } finally {
            this.close();
        }
    }

    private String label() {
        return this.deviceId == null ? this.remoteAddress() : this.deviceId;
    }

    /* ------------------------------------------------------------------ handshake */

    private boolean authenticate(DataInputStream in) throws IOException {
        byte[] random = new byte[16];
        new SecureRandom().nextBytes(random);
        String nonce = HexFormat.of().formatHex(random);
        this.sendJson(Protocol.CHALLENGE, new JSONObject().put("nonce", nonce).put("proto", Protocol.VERSION));

        this.socket.setSoTimeout(10_000);
        int type = in.readUnsignedByte();
        int length = in.readInt();
        if (type != Protocol.HELLO || length <= 0 || length > 2048) {
            this.plugin.getPluginLogger().warning("Voice connection from " + this.remoteAddress() + " sent no valid hello");
            return false;
        }
        byte[] payload = new byte[length];
        in.readFully(payload);
        this.socket.setSoTimeout(SOCKET_TIMEOUT_MS);
        JSONObject hello = new JSONObject(new String(payload, StandardCharsets.UTF_8));
        String id = hello.optString("id", "").trim();
        if (!id.matches("[A-Za-z0-9_-]{3,48}")) {
            this.sendJson(Protocol.ERROR, new JSONObject().put("message", "invalid device id"));
            return false;
        }
        String expected = hmac(this.config.deviceKey, nonce + ":" + id);
        String given = hello.optString("hmac", "");
        if (!MessageDigest.isEqual(expected.getBytes(StandardCharsets.US_ASCII), given.toLowerCase().getBytes(StandardCharsets.US_ASCII))) {
            this.plugin.getPluginLogger().warning("Voice device " + id + " from " + this.remoteAddress() + " has a wrong device key");
            this.sendJson(Protocol.ERROR, new JSONObject().put("message", "authentication failed"));
            return false;
        }
        this.deviceId = id;
        this.firmware = hello.optString("fw", "");
        this.deviceName = this.plugin.registerDevice(id, hello.optString("name", ""), this);
        this.accessState = this.plugin.accessState(id, this.deviceName);
        this.sendJson(Protocol.WELCOME, new JSONObject().put("name", this.deviceName).put("state", this.accessState));
        this.plugin.getPluginLogger().info("Voice device " + id + " (" + this.deviceName + ", fw " + this.firmware + ") connected from "
                + this.remoteAddress());
        return true;
    }

    static String hmac(String key, String message) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(key.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
            return HexFormat.of().formatHex(mac.doFinal(message.getBytes(StandardCharsets.UTF_8)));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /* ------------------------------------------------------------------ incoming frames */

    private void handle(int type, byte[] payload) throws IOException {
        switch (type) {
            case Protocol.PING -> {
                this.send(Protocol.PONG, new byte[0]);
                this.refreshAccess();
            }
            case Protocol.AUDIO -> this.onAudio(payload);
            case Protocol.EVENT -> this.onEvent(new JSONObject(new String(payload, StandardCharsets.UTF_8)));
            default -> this.plugin.getPluginLogger().debug("Voice device " + this.deviceId + " sent unknown frame " + type);
        }
    }

    /**
     * A device waiting for approval learns about the decision with the next ping.
     */
    private void refreshAccess() throws IOException {
        if (this.mode != Mode.IDLE) {
            return;
        }
        String state = this.plugin.accessState(this.deviceId, this.deviceName);
        if (!state.equals(this.accessState)) {
            this.accessState = state;
            this.sendJson(Protocol.STATE, new JSONObject().put("state", state));
        }
    }

    private void onEvent(JSONObject event) throws IOException {
        String name = event.optString("event", "");
        switch (name) {
            case "listen_start" -> {
                this.generation.incrementAndGet();
                this.detector = new SpeechDetector(this.config);
                this.recording = new ByteArrayOutputStream();
                this.mode = Mode.LISTENING;
            }
            case "listen_stop" -> {
                if (this.mode == Mode.LISTENING) {
                    this.finishRecording(this.detector != null && this.detector.durationMs() >= 400);
                }
            }
            case "cancel" -> {
                this.generation.incrementAndGet();
                this.mode = Mode.IDLE;
                this.recording = null;
                this.sendJson(Protocol.PLAY_END, new JSONObject().put("cancelled", true));
            }
            case "mute" -> this.muted = event.optBoolean("value", false);
            case "playback_done" -> {
                if (this.mode == Mode.SPEAKING) {
                    this.mode = Mode.IDLE;
                }
            }
            default -> this.plugin.getPluginLogger().debug("Voice device " + this.deviceId + " sent unknown event " + name);
        }
    }

    private void onAudio(byte[] pcm) throws IOException {
        if (this.mode != Mode.LISTENING || this.recording == null) {
            return;
        }
        if (this.recording.size() + pcm.length > MAX_RECORDING_BYTES) {
            this.finishRecording(true);
            return;
        }
        this.recording.writeBytes(pcm);
        short[] samples = Audio.toSamples(pcm);
        switch (this.detector.feed(samples, samples.length)) {
            case END_OF_SPEECH, TOO_LONG -> this.finishRecording(true);
            case NO_SPEECH -> this.finishRecording(false);
            default -> {
            }
        }
    }

    /**
     * End the recording on the device and start the turn if there is something to understand.
     */
    private void finishRecording(boolean understand) throws IOException {
        byte[] pcm = this.recording == null ? new byte[0] : this.recording.toByteArray();
        this.recording = null;
        this.send(Protocol.STOP_LISTEN, new byte[0]);
        if (!understand || pcm.length == 0) {
            this.mode = Mode.IDLE;
            this.sendJson(Protocol.STATE, new JSONObject().put("state", "idle"));
            return;
        }
        this.mode = Mode.THINKING;
        this.sendJson(Protocol.STATE, new JSONObject().put("state", "thinking"));
        int turn = this.generation.incrementAndGet();
        byte[] wav = Audio.wav(pcm, Protocol.SAMPLE_RATE);
        this.workers.submit(() -> this.runTurn(turn, wav));
    }

    /* ------------------------------------------------------------------ turn */

    private void runTurn(int turn, byte[] wav) {
        TURN.set(this);
        try {
            MirraVoice.Answer answer = this.plugin.getMirraVoice().handle(this.deviceId, this.deviceName, wav);
            if (turn != this.generation.get() || this.closed) {
                return;
            }
            switch (answer.state()) {
                case "ANSWER", "PAUSED" -> {
                    if (answer.hasAnswer()) {
                        boolean followUp = this.config.followUpOnQuestion && answer.answer().trim().endsWith("?");
                        this.speak(answer.answer(), turn, followUp);
                    } else {
                        this.finishTurn(turn, "idle");
                    }
                }
                case "PENDING" -> {
                    this.accessState = "pending";
                    this.finishTurn(turn, "pending");
                }
                case "DENIED" -> {
                    this.accessState = "denied";
                    this.finishTurn(turn, "denied");
                }
                case "EMPTY" -> this.finishTurn(turn, "idle");
                default -> this.finishTurn(turn, "error");
            }
        } catch (Exception e) {
            this.plugin.getPluginLogger().error("Voice turn of " + this.deviceId + " failed", e);
            this.finishTurn(turn, "error");
        } finally {
            TURN.remove();
        }
    }

    private void finishTurn(int turn, String state) {
        if (turn != this.generation.get()) {
            return;
        }
        this.mode = Mode.IDLE;
        try {
            this.sendJson(Protocol.STATE, new JSONObject().put("state", state));
        } catch (IOException e) {
            this.close();
        }
    }

    /**
     * Announcement outside of a conversation: waits until the device is free, then speaks.
     */
    public void announce(String text) throws Exception {
        if (TURN.get() == this) {
            this.speak(text, this.generation.get(), false);
            return;
        }
        long deadline = System.currentTimeMillis() + 30_000;
        while ((this.mode == Mode.LISTENING || this.mode == Mode.THINKING) && System.currentTimeMillis() < deadline && this.isOpen()) {
            Thread.sleep(200);
        }
        if (!this.isOpen()) {
            throw new IOException("Device is offline");
        }
        this.speak(text, this.generation.get(), false);
    }

    /**
     * Synthesize and stream a text. The text is split into sentences so the first one plays while
     * the rest is still being synthesized.
     */
    private void speak(String text, int turn, boolean listenAfter) throws Exception {
        List<String> chunks = chunks(clean(text), this.config.speechChunkChars);
        if (chunks.isEmpty()) {
            this.finishTurn(turn, "idle");
            return;
        }
        List<CompletableFuture<byte[]>> audio = new ArrayList<>();
        for (String chunk : chunks) {
            audio.add(CompletableFuture.supplyAsync(() -> {
                try {
                    short[] speech = Audio.toSamples(this.plugin.getMirraVoice().synthesize(chunk));
                    return Audio.toBytes(Audio.resample(speech, MirraVoice.SPEECH_SAMPLE_RATE, Protocol.SAMPLE_RATE));
                } catch (Exception e) {
                    throw new IllegalStateException(e.getMessage(), e);
                }
            }, this.workers));
        }
        this.speechLock.lockInterruptibly();
        try {
            boolean started = false;
            for (CompletableFuture<byte[]> part : audio) {
                byte[] pcm;
                try {
                    pcm = part.get(60, TimeUnit.SECONDS);
                } catch (Exception e) {
                    this.plugin.getPluginLogger().warning("Speech synthesis for " + this.deviceId + " failed: " + e.getMessage());
                    if (!started) {
                        this.finishTurn(turn, "error");
                        return;
                    }
                    break;
                }
                for (int offset = 0; offset < pcm.length; offset += PLAY_FRAME_BYTES) {
                    if (turn != this.generation.get() || this.closed) {
                        audio.forEach(future -> future.cancel(true));
                        return;
                    }
                    int length = Math.min(PLAY_FRAME_BYTES, pcm.length - offset);
                    byte[] frame = new byte[length];
                    System.arraycopy(pcm, offset, frame, 0, length);
                    this.send(Protocol.PLAY, frame);
                    if (!started) {
                        started = true;
                        this.mode = Mode.SPEAKING;
                    }
                }
            }
            if (turn == this.generation.get()) {
                this.sendJson(Protocol.PLAY_END, new JSONObject().put("listen", listenAfter && !this.muted));
            }
        } finally {
            this.speechLock.unlock();
        }
    }

    static String clean(String text) {
        String spoken = NOT_SPOKEN.matcher(text).replaceAll(" ");
        /* links are not read out */
        spoken = spoken.replaceAll("https?://\\S+", "");
        return spoken.replaceAll("\\s+", " ").replaceAll(" ([.,!?;:])", "$1").trim();
    }

    static List<String> chunks(String text, int maxChars) {
        List<String> chunks = new ArrayList<>();
        if (text.isBlank()) {
            return chunks;
        }
        StringBuilder current = new StringBuilder();
        for (String sentence : SENTENCE_END.split(text)) {
            /* the first chunk is kept short so the answer starts quickly */
            int limit = chunks.isEmpty() ? Math.min(120, maxChars) : maxChars;
            if (current.length() > 0 && current.length() + sentence.length() + 1 > limit) {
                chunks.add(current.toString());
                current.setLength(0);
            }
            if (current.length() > 0) {
                current.append(' ');
            }
            current.append(sentence);
        }
        if (current.length() > 0) {
            chunks.add(current.toString());
        }
        return chunks;
    }

    /* ------------------------------------------------------------------ output */

    public void command(JSONObject command) throws IOException {
        this.sendJson(Protocol.CMD, command);
    }

    private void sendJson(int type, JSONObject json) throws IOException {
        this.send(type, json.toString().getBytes(StandardCharsets.UTF_8));
    }

    private void send(int type, byte[] payload) throws IOException {
        synchronized (this.writeLock) {
            byte[] header = {(byte) type, (byte) (payload.length >>> 24), (byte) (payload.length >>> 16),
                    (byte) (payload.length >>> 8), (byte) payload.length};
            this.out.write(header);
            this.out.write(payload);
            this.out.flush();
        }
    }

    public void close() {
        if (this.closed) {
            return;
        }
        this.closed = true;
        this.generation.incrementAndGet();
        try {
            this.socket.close();
        } catch (IOException ignored) {
            /* closing anyway */
        }
        this.plugin.unregisterConnection(this);
    }
}
