// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { v4 as uuid } from 'uuid';

import { fetchClient } from '../../api/client';

export type WebRTCConnectionStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'failed';

type WebRTCConnectionEvent =
    | {
          type: 'status_change';
          status: WebRTCConnectionStatus;
      }
    | {
          type: 'error';
          error: Error;
      };

export type Listener = (event: WebRTCConnectionEvent) => void;

type SessionData =
    | RTCSessionDescriptionInit
    | {
          status: 'failed';
          meta: { error: 'concurrency_limit_reached'; limit: number };
      };

const CONNECTION_TIMEOUT = 5000;
const CLOSE_CONNECTION_DELAY = 500;

export class WebRTCConnection {
    private peerConnection: RTCPeerConnection | undefined;
    private webrtcId: string;
    private status: WebRTCConnectionStatus = 'idle';
    private dataChannel: RTCDataChannel | undefined;

    private listeners: Array<Listener> = [];
    private timeoutId?: ReturnType<typeof setTimeout>;

    constructor() {
        this.webrtcId = uuid();
    }

    public getStatus(): WebRTCConnectionStatus {
        return this.status;
    }

    public getPeerConnection(): RTCPeerConnection | undefined {
        return this.peerConnection;
    }

    public getId(): string {
        return this.webrtcId;
    }

    public async start(): Promise<void> {
        if (this.hasActiveConnection()) {
            console.warn('WebRTC connection is already active or in progress.');
            return;
        }

        this.updateStatus('connecting');
        const iceServers = await this.fetchIceServers();
        const rtcConfig: RTCConfiguration = {
            iceServers,
        };
        this.peerConnection = new RTCPeerConnection(rtcConfig);
        this.timeoutId = setTimeout(() => {
            console.warn('Connection is taking longer than usual. Are you on a VPN?');
        }, CONNECTION_TIMEOUT);

        try {
            this.setupPeerConnection();

            await this.createAndSetOffer();
            try {
                await this.waitForIceGathering();
            } catch (e) {
                console.warn('ICE gathering timed out or failed, proceeding anyway:', e);
            }
            const data = await this.sendOffer();

            if (!this.handleOfferResponse(data)) return;

            this.setupConnectionStateListener();
        } catch (err) {
            clearTimeout(this.timeoutId);
            console.error('Error setting up WebRTC:', err);

            this.emit({ type: 'error', error: err as Error });
            this.updateStatus('failed');

            await this.stop();
        }

        if (this.peerConnection) {
            this.peerConnection.getTransceivers().forEach((t) => (t.direction = 'recvonly'));
        }
    }

    private setupPeerConnection() {
        if (!this.peerConnection) return;

        this.peerConnection.addTransceiver('video', { direction: 'recvonly' });
        this.dataChannel = this.peerConnection.createDataChannel('text');
        this.dataChannel.onopen = () => {
            this.dataChannel?.send('handshake');
        };
    }

    private async createAndSetOffer() {
        if (!this.peerConnection) return;

        const offer = await this.peerConnection.createOffer();

        await this.peerConnection.setLocalDescription(offer);
    }

    private async waitForIceGathering(): Promise<void> {
        await Promise.race([
            new Promise<void>((resolve) => {
                if (!this.peerConnection || this.peerConnection.iceGatheringState === 'complete') {
                    resolve();
                    return;
                }

                const checkState = () => {
                    if (this.peerConnection && this.peerConnection.iceGatheringState === 'complete') {
                        this.peerConnection.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };

                this.peerConnection?.addEventListener('icegatheringstatechange', checkState);
            }),
            new Promise<void>((_, reject) =>
                setTimeout(() => reject(new Error('ICE gathering timed out')), CONNECTION_TIMEOUT)
            ),
        ]);
    }

    private async sendOffer(): Promise<SessionData | undefined> {
        if (!this.peerConnection) return;

        const { data } = await fetchClient.POST('/api/webrtc/offer', {
            body: {
                sdp: this.peerConnection.localDescription?.sdp ?? '',
                type: this.peerConnection.localDescription?.type ?? '',
                webrtc_id: this.webrtcId,
            },
        });

        return data as SessionData;
    }

    private async handleOfferResponse(data: SessionData | undefined): Promise<boolean> {
        if (!data) return false;

        if ('status' in data && data.status === 'failed') {
            const errorMessage =
                data.meta.error === 'concurrency_limit_reached'
                    ? `Too many connections. Maximum limit is ${data.meta.limit}`
                    : data.meta.error;

            console.error(errorMessage);

            this.emit({ type: 'error', error: new Error(errorMessage) });

            return false;
        }

        if (this.peerConnection) {
            await this.peerConnection.setRemoteDescription(data as RTCSessionDescriptionInit);
        }

        return true;
    }

    private setupConnectionStateListener() {
        if (!this.peerConnection) return;

        this.peerConnection.addEventListener('connectionstatechange', () => {
            if (!this.peerConnection) return;

            switch (this.peerConnection.connectionState) {
                case 'connected':
                    this.updateStatus('connected');
                    clearTimeout(this.timeoutId);
                    break;
                case 'disconnected':
                    this.updateStatus('disconnected');
                    break;
                case 'failed':
                    this.updateStatus('failed');
                    this.emit({ type: 'error', error: new Error('WebRTC connection failed.') });
                    break;
                case 'closed':
                    this.updateStatus('disconnected');
                    break;
                default:
                    this.updateStatus('connecting');
                    break;
            }
        });
    }

    public async stop(): Promise<void> {
        if (!this.peerConnection) {
            return;
        }

        const transceivers = this.peerConnection.getTransceivers();

        transceivers.forEach((transceiver) => {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });

        const senders = this.peerConnection.getSenders();

        senders.forEach((sender) => {
            if (sender.track && sender.track.stop) sender.track.stop();
        });

        // Give a brief moment for tracks to stop before closing the connection
        await new Promise<void>((resolve) =>
            setTimeout(() => {
                if (this.peerConnection) {
                    this.peerConnection.close();
                    this.peerConnection = undefined;
                    this.updateStatus('idle');
                }

                resolve();
            }, CLOSE_CONNECTION_DELAY)
        );
    }

    public subscribe(listener: Listener): () => void {
        this.listeners.push(listener);

        return () => this.unsubscribe(listener);
    }

    private hasActiveConnection(): boolean {
        return this.peerConnection !== undefined && this.status !== 'idle' && this.status !== 'disconnected';
    }

    private unsubscribe(listener: Listener) {
        this.listeners = this.listeners.filter((currentListener) => currentListener !== listener);
    }

    private emit(event: WebRTCConnectionEvent) {
        this.listeners.forEach((listener) => listener(event));
    }

    private updateStatus(newStatus: WebRTCConnectionStatus) {
        if (this.status !== newStatus) {
            this.status = newStatus;
            this.emit({ type: 'status_change', status: newStatus });
        }
    }

    private updateConfThreshold(value: number) {
        return fetchClient.POST('/api/webrtc/input_hook', {
            body: {
                conf_threshold: value,
                webrtc_id: this.webrtcId,
            },
        });
    }

    private async fetchIceServers() {
        try {
            const { data } = await fetchClient.GET('/api/webrtc/config');
            if (!data) {
                console.warn('No ICE server data received.');
                return [];
            }
            console.info('ICE servers:', data.iceServers);
            return data.iceServers as RTCIceServer[];
        } catch (err) {
            console.error('Error fetching ICE servers:', err);
            return [];
        }
    }
}
