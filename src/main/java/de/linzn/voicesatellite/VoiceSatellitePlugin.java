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

import de.linzn.mirra.MirraPlugin;
import de.linzn.mirra.voice.MirraVoice;
import de.linzn.stem.modules.pluginModule.STEMPlugin;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Voice satellites (Onju V3 boards in Google Home Mini shells) as speaking and listening devices for Mirra.
 */
public class VoiceSatellitePlugin extends STEMPlugin {

    public static VoiceSatellitePlugin voiceSatellitePlugin;

    private final Map<String, SatelliteConnection> connections = new ConcurrentHashMap<>();
    private SatelliteConfig config;
    private SatelliteServer server;
    private ExecutorService workers;
    private MirraVoice.Speaker speaker;

    public VoiceSatellitePlugin() {
        voiceSatellitePlugin = this;
    }

    @Override
    public void onEnable() {
        this.config = new SatelliteConfig(this.getDefaultConfig());
        if (this.config.generatedKey) {
            this.getPluginLogger().warning("New device key generated (server.deviceKey in the config of voice-satellite): "
                    + this.config.deviceKey + " - put it into config.h of the firmware");
        }
        this.workers = Executors.newThreadPerTaskExecutor(Thread.ofVirtual().name("voice-satellite-work-", 0).factory());
        this.speaker = new MirraVoice.Speaker() {
            @Override
            public List<MirraVoice.Device> devices() {
                return VoiceSatellitePlugin.this.devices();
            }

            @Override
            public void say(String deviceId, String text) throws Exception {
                SatelliteConnection connection = VoiceSatellitePlugin.this.connection(deviceId)
                        .orElseThrow(() -> new IOException("Device " + deviceId + " is offline"));
                connection.announce(text);
            }
        };
        this.getMirraVoice().registerSpeaker(this.speaker);
        this.getStemApi().commands().register("voice", new VoiceCommand(this));
        this.server = new SatelliteServer(this, this.config, this.workers);
        try {
            this.server.start();
        } catch (IOException e) {
            this.getPluginLogger().error("Could not open port " + this.config.port + " for voice satellites", e);
        }
    }

    @Override
    public void onDisable() {
        if (this.speaker != null && this.getMirraVoice() != null) {
            this.getMirraVoice().unregisterSpeaker(this.speaker);
        }
        if (this.server != null) {
            this.server.stop();
        }
        new ArrayList<>(this.connections.values()).forEach(SatelliteConnection::close);
        if (this.workers != null) {
            this.workers.shutdownNow();
        }
    }

    public MirraVoice getMirraVoice() {
        return MirraPlugin.mirraPlugin == null ? null : MirraPlugin.mirraPlugin.getVoice();
    }

    /**
     * Remember a device that authenticated. A name in the config wins over the name the device sends.
     *
     * @return the room name of the device
     */
    String registerDevice(String id, String announcedName, SatelliteConnection connection) {
        SatelliteConnection previous = this.connections.put(id, connection);
        if (previous != null && previous != connection) {
            previous.close();
        }
        String path = "devices." + id + ".name";
        Object configured = this.getDefaultConfig().get(path, announcedName == null || announcedName.isBlank() ? id : announcedName);
        this.getDefaultConfig().save();
        return String.valueOf(configured);
    }

    void unregisterConnection(SatelliteConnection connection) {
        if (connection.deviceId() != null) {
            this.connections.remove(connection.deviceId(), connection);
        }
    }

    String accessState(String deviceId, String deviceName) {
        MirraVoice voice = this.getMirraVoice();
        return voice == null ? "idle" : voice.accessState(deviceId);
    }

    int connectionCount() {
        return this.connections.size();
    }

    Optional<SatelliteConnection> connection(String idOrName) {
        for (SatelliteConnection connection : this.connections.values()) {
            if (connection.deviceId().equalsIgnoreCase(idOrName) || idOrName.equalsIgnoreCase(connection.deviceName())) {
                return Optional.of(connection);
            }
        }
        return Optional.empty();
    }

    /**
     * Devices known from the config, online or not.
     */
    List<MirraVoice.Device> devices() {
        List<MirraVoice.Device> devices = new ArrayList<>();
        Map<String, Boolean> seen = new ConcurrentHashMap<>();
        for (SatelliteConnection connection : this.connections.values()) {
            if (connection.deviceId() != null && connection.isOpen()) {
                devices.add(new MirraVoice.Device(connection.deviceId(), connection.deviceName(), true));
                seen.put(connection.deviceId(), true);
            }
        }
        Object section = this.getDefaultConfig().get("devices", null);
        if (section instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                String id = String.valueOf(entry.getKey());
                if (!seen.containsKey(id)) {
                    Object name = entry.getValue() instanceof Map<?, ?> values ? values.get("name") : null;
                    devices.add(new MirraVoice.Device(id, name == null ? id : String.valueOf(name), false));
                }
            }
        }
        return devices;
    }

    void sendCommand(String idOrName, JSONObject command) throws IOException {
        SatelliteConnection connection = this.connection(idOrName).orElseThrow(() -> new IOException("Device " + idOrName + " is offline"));
        connection.command(command);
    }

    List<SatelliteConnection> connections() {
        return new ArrayList<>(this.connections.values());
    }
}
