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

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.concurrent.ExecutorService;

/**
 * TCP server the satellites connect to. Each connection gets its own thread.
 */
public final class SatelliteServer implements Runnable {

    private final VoiceSatellitePlugin plugin;
    private final SatelliteConfig config;
    private final ExecutorService workers;
    private ServerSocket serverSocket;
    private Thread thread;
    private volatile boolean running;

    public SatelliteServer(VoiceSatellitePlugin plugin, SatelliteConfig config, ExecutorService workers) {
        this.plugin = plugin;
        this.config = config;
        this.workers = workers;
    }

    public void start() throws IOException {
        this.serverSocket = new ServerSocket();
        this.serverSocket.setReuseAddress(true);
        this.serverSocket.bind(new InetSocketAddress(InetAddress.getByName(this.config.bindAddress), this.config.port));
        this.running = true;
        this.thread = new Thread(this, "voice-satellite-server");
        this.thread.setDaemon(true);
        this.thread.start();
        this.plugin.getPluginLogger().info("Voice satellite server listening on " + this.config.bindAddress + ":" + this.config.port);
    }

    @Override
    public void run() {
        while (this.running) {
            try {
                Socket socket = this.serverSocket.accept();
                InetAddress address = socket.getInetAddress();
                if (this.config.onlyPrivateNetworks && !isPrivate(address)) {
                    this.plugin.getPluginLogger().warning("Voice connection from public address " + address.getHostAddress() + " refused");
                    socket.close();
                    continue;
                }
                if (this.plugin.connectionCount() >= this.config.maxDevices) {
                    this.plugin.getPluginLogger().warning("Voice connection from " + address.getHostAddress() + " refused, too many devices");
                    socket.close();
                    continue;
                }
                SatelliteConnection connection = new SatelliteConnection(this.plugin, this.config, socket, this.workers);
                Thread worker = new Thread(connection, "voice-satellite-" + address.getHostAddress());
                worker.setDaemon(true);
                worker.start();
            } catch (SocketException e) {
                if (this.running) {
                    this.plugin.getPluginLogger().error("Voice satellite server failed", e);
                }
            } catch (IOException e) {
                this.plugin.getPluginLogger().warning("Accepting a voice connection failed: " + e.getMessage());
            }
        }
    }

    static boolean isPrivate(InetAddress address) {
        return address.isLoopbackAddress() || address.isSiteLocalAddress() || address.isLinkLocalAddress()
                || (address.getAddress().length == 16 && (address.getAddress()[0] & 0xFE) == 0xFC);
    }

    public void stop() {
        this.running = false;
        try {
            if (this.serverSocket != null) {
                this.serverSocket.close();
            }
        } catch (IOException ignored) {
            /* shutting down */
        }
    }
}
