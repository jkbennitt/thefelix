/**
 * Felix Framework - Real-Time Streaming Components
 *
 * Type-safe real-time data streaming for agent movement, performance metrics,
 * and system updates with WebSocket support, efficient batching, and
 * mobile-optimized data delivery.
 *
 * Features:
 * - WebSocket-based real-time updates
 * - Efficient data batching and compression
 * - Agent position tracking with interpolation
 * - Performance metrics streaming
 * - Mobile-optimized update rates
 * - Automatic reconnection and error recovery
 * - Memory-efficient circular buffers
 *
 * @version 1.0.0
 * @author Felix Framework Team
 */

import {
  RealTimeUpdate,
  AgentPositionUpdate,
  PerformanceUpdate,
  VisualizationUpdateBatch,
  ProgressTracker,
  StatusIndicator,
  MobileOptimizations
} from '../types/gradio-interface';

import {
  Position3D,
  AgentInstance,
  AgentState,
  PerformanceMetrics,
  FelixMessage,
  MessageType,
  Coordinate3D
} from '../types/felix-core';

// =============================================================================
// Streaming Configuration Types
// =============================================================================

/** Real-time streaming configuration */
export interface StreamingConfig {
  readonly websocketUrl: string;
  readonly updateInterval: number; // milliseconds
  readonly maxBufferSize: number;
  readonly compressionEnabled: boolean;
  readonly batchSize: number;
  readonly reconnectAttempts: number;
  readonly heartbeatInterval: number;
  readonly mobileOptimizations: MobileOptimizations;
}

/** Stream subscription options */
export interface StreamSubscription {
  readonly id: string;
  readonly type: StreamType;
  readonly filters?: StreamFilters;
  readonly callback: StreamCallback<unknown>;
  readonly priority: StreamPriority;
}

/** Stream types */
export enum StreamType {
  AgentPositions = 'agent_positions',
  PerformanceMetrics = 'performance_metrics',
  SystemStatus = 'system_status',
  TaskProgress = 'task_progress',
  ErrorReports = 'error_reports',
  All = 'all'
}

/** Stream filtering options */
export interface StreamFilters {
  readonly agentIds?: readonly string[];
  readonly agentTypes?: readonly string[];
  readonly metricNames?: readonly string[];
  readonly minPriority?: StreamPriority;
  readonly maxUpdateRate?: number;
}

/** Stream callback function */
export type StreamCallback<T> = (data: T, metadata: StreamMetadata) => void | Promise<void>;

/** Stream priority levels */
export enum StreamPriority {
  Critical = 0,
  High = 1,
  Medium = 2,
  Low = 3,
  Background = 4
}

/** Stream metadata */
export interface StreamMetadata {
  readonly timestamp: number;
  readonly sequence: number;
  readonly size: number;
  readonly compression: string;
  readonly latency: number;
}

// =============================================================================
// Data Buffer Types
// =============================================================================

/** Circular buffer for efficient data storage */
export class CircularBuffer<T> {
  private buffer: (T | undefined)[];
  private head: number = 0;
  private tail: number = 0;
  private count: number = 0;

  constructor(private readonly capacity: number) {
    this.buffer = new Array(capacity);
  }

  push(item: T): void {
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;

    if (this.count < this.capacity) {
      this.count++;
    } else {
      this.head = (this.head + 1) % this.capacity;
    }
  }

  getLatest(n: number = 1): T[] {
    const result: T[] = [];
    const available = Math.min(n, this.count);

    for (let i = 0; i < available; i++) {
      const index = (this.tail - 1 - i + this.capacity) % this.capacity;
      const item = this.buffer[index];
      if (item !== undefined) {
        result.unshift(item);
      }
    }

    return result;
  }

  getAll(): T[] {
    const result: T[] = [];

    for (let i = 0; i < this.count; i++) {
      const index = (this.head + i) % this.capacity;
      const item = this.buffer[index];
      if (item !== undefined) {
        result.push(item);
      }
    }

    return result;
  }

  clear(): void {
    this.buffer.fill(undefined);
    this.head = 0;
    this.tail = 0;
    this.count = 0;
  }

  get size(): number {
    return this.count;
  }

  get capacity(): number {
    return this.capacity;
  }
}

/** Agent position history with interpolation */
export interface AgentPositionHistory {
  readonly agentId: string;
  readonly positions: CircularBuffer<TimestampedPosition>;
  readonly interpolator: PositionInterpolator;
}

/** Position with timestamp for interpolation */
export interface TimestampedPosition {
  readonly position: Position3D;
  readonly timestamp: number;
  readonly velocity?: Position3D;
  readonly helixParameter: number;
}

/** Position interpolation utility */
export class PositionInterpolator {
  /**
   * Interpolate position between two timestamped positions
   */
  static interpolate(
    pos1: TimestampedPosition,
    pos2: TimestampedPosition,
    targetTime: number
  ): Position3D {
    if (targetTime <= pos1.timestamp) return pos1.position;
    if (targetTime >= pos2.timestamp) return pos2.position;

    const timeDiff = pos2.timestamp - pos1.timestamp;
    const timeRatio = (targetTime - pos1.timestamp) / timeDiff;

    // Linear interpolation
    const x = this.lerp(pos1.position.x, pos2.position.x, timeRatio);
    const y = this.lerp(pos1.position.y, pos2.position.y, timeRatio);
    const z = this.lerp(pos1.position.z, pos2.position.z, timeRatio);

    return {
      x: x as Coordinate3D,
      y: y as Coordinate3D,
      z: z as Coordinate3D
    };
  }

  /**
   * Smooth interpolation using velocity if available
   */
  static smoothInterpolate(
    pos1: TimestampedPosition,
    pos2: TimestampedPosition,
    targetTime: number
  ): Position3D {
    if (!pos1.velocity || !pos2.velocity) {
      return this.interpolate(pos1, pos2, targetTime);
    }

    // Hermite interpolation for smooth movement
    const timeDiff = pos2.timestamp - pos1.timestamp;
    const t = (targetTime - pos1.timestamp) / timeDiff;

    const h1 = 2 * t * t * t - 3 * t * t + 1;
    const h2 = -2 * t * t * t + 3 * t * t;
    const h3 = t * t * t - 2 * t * t + t;
    const h4 = t * t * t - t * t;

    const x = h1 * pos1.position.x + h2 * pos2.position.x +
              h3 * pos1.velocity.x * timeDiff + h4 * pos2.velocity.x * timeDiff;
    const y = h1 * pos1.position.y + h2 * pos2.position.y +
              h3 * pos1.velocity.y * timeDiff + h4 * pos2.velocity.y * timeDiff;
    const z = h1 * pos1.position.z + h2 * pos2.position.z +
              h3 * pos1.velocity.z * timeDiff + h4 * pos2.velocity.z * timeDiff;

    return {
      x: x as Coordinate3D,
      y: y as Coordinate3D,
      z: z as Coordinate3D
    };
  }

  private static lerp(start: number, end: number, t: number): number {
    return start + (end - start) * t;
  }
}

// =============================================================================
// Real-Time Streaming Manager
// =============================================================================

export class RealTimeStreamingManager {
  private websocket: WebSocket | null = null;
  private subscriptions: Map<string, StreamSubscription> = new Map();
  private agentHistories: Map<string, AgentPositionHistory> = new Map();
  private performanceBuffer: CircularBuffer<PerformanceUpdate> = new CircularBuffer(1000);
  private statusBuffer: CircularBuffer<StatusIndicator> = new CircularBuffer(100);

  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private updateTimer: NodeJS.Timeout | null = null;

  private connectionState: ConnectionState = ConnectionState.Disconnected;
  private reconnectAttempts: number = 0;
  private lastHeartbeat: number = 0;
  private sequenceNumber: number = 0;

  constructor(private readonly config: StreamingConfig) {
    this.initializeBuffers();
  }

  // =============================================================================
  // Connection Management
  // =============================================================================

  /**
   * Initialize WebSocket connection
   */
  public async connect(): Promise<void> {
    if (this.connectionState === ConnectionState.Connected ||
        this.connectionState === ConnectionState.Connecting) {
      return;
    }

    this.connectionState = ConnectionState.Connecting;

    try {
      this.websocket = new WebSocket(this.config.websocketUrl);
      this.setupWebSocketHandlers();

      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('WebSocket connection timeout'));
        }, 10000);

        this.websocket!.onopen = () => {
          clearTimeout(timeout);
          this.connectionState = ConnectionState.Connected;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.startUpdateLoop();
          resolve();
        };

        this.websocket!.onerror = (error) => {
          clearTimeout(timeout);
          reject(error);
        };
      });
    } catch (error) {
      this.connectionState = ConnectionState.Error;
      throw error;
    }
  }

  /**
   * Disconnect from WebSocket
   */
  public disconnect(): void {
    this.connectionState = ConnectionState.Disconnecting;

    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }

    this.stopTimers();
    this.connectionState = ConnectionState.Disconnected;
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupWebSocketHandlers(): void {
    if (!this.websocket) return;

    this.websocket.onmessage = (event) => {
      this.handleMessage(event);
    };

    this.websocket.onclose = (event) => {
      this.handleDisconnection(event);
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.connectionState = ConnectionState.Error;
    };
  }

  /**
   * Handle incoming WebSocket messages
   */
  private async handleMessage(event: MessageEvent): Promise<void> {
    try {
      const data = JSON.parse(event.data) as RealTimeUpdate;

      // Update sequence tracking
      this.sequenceNumber = Math.max(this.sequenceNumber, data.sequence);

      // Route message to appropriate handlers
      switch (data.type) {
        case 'agent_position':
          await this.handleAgentPositionUpdate(data.data as AgentPositionUpdate);
          break;
        case 'performance_metrics':
          await this.handlePerformanceUpdate(data.data as PerformanceUpdate);
          break;
        case 'system_status':
          await this.handleStatusUpdate(data.data as StatusIndicator);
          break;
        case 'heartbeat':
          this.lastHeartbeat = Date.now();
          break;
        default:
          console.warn('Unknown message type:', data.type);
      }

      // Notify subscribers
      await this.notifySubscribers(data);

    } catch (error) {
      console.error('Error handling message:', error);
    }
  }

  /**
   * Handle WebSocket disconnection
   */
  private handleDisconnection(event: CloseEvent): void {
    this.connectionState = ConnectionState.Disconnected;
    this.stopTimers();

    if (event.code !== 1000 && this.reconnectAttempts < this.config.reconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  /**
   * Schedule automatic reconnection
   */
  private scheduleReconnect(): void {
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        console.error('Reconnection failed:', error);
        this.scheduleReconnect();
      }
    }, delay);
  }

  // =============================================================================
  // Data Handling
  // =============================================================================

  /**
   * Handle agent position updates
   */
  private async handleAgentPositionUpdate(update: AgentPositionUpdate): Promise<void> {
    let history = this.agentHistories.get(update.agentId);

    if (!history) {
      history = {
        agentId: update.agentId,
        positions: new CircularBuffer<TimestampedPosition>(100),
        interpolator: PositionInterpolator
      };
      this.agentHistories.set(update.agentId, history);
    }

    // Calculate velocity from previous position
    const previousPositions = history.positions.getLatest(1);
    let velocity: Position3D | undefined;

    if (previousPositions.length > 0) {
      const prev = previousPositions[0];
      const timeDiff = update.timestamp - prev.timestamp;

      if (timeDiff > 0) {
        velocity = {
          x: ((update.position.x - prev.position.x) / timeDiff) as Coordinate3D,
          y: ((update.position.y - prev.position.y) / timeDiff) as Coordinate3D,
          z: ((update.position.z - prev.position.z) / timeDiff) as Coordinate3D
        };
      }
    }

    const timestampedPosition: TimestampedPosition = {
      position: update.position,
      timestamp: update.timestamp,
      velocity,
      helixParameter: update.helixParameter
    };

    history.positions.push(timestampedPosition);
  }

  /**
   * Handle performance metric updates
   */
  private async handlePerformanceUpdate(update: PerformanceUpdate): Promise<void> {
    this.performanceBuffer.push(update);

    // Check for performance alerts
    if (update.metrics.errorRate > 0.1) {
      this.statusBuffer.push({
        type: 'warning' as any,
        message: `High error rate detected: ${(update.metrics.errorRate * 100).toFixed(1)}%`,
        timestamp: update.timestamp
      });
    }

    if (update.metrics.latency > 5000) {
      this.statusBuffer.push({
        type: 'warning' as any,
        message: `High latency detected: ${update.metrics.latency}ms`,
        timestamp: update.timestamp
      });
    }
  }

  /**
   * Handle system status updates
   */
  private async handleStatusUpdate(status: StatusIndicator): Promise<void> {
    this.statusBuffer.push(status);
  }

  // =============================================================================
  // Subscription Management
  // =============================================================================

  /**
   * Subscribe to real-time updates
   */
  public subscribe<T>(
    type: StreamType,
    callback: StreamCallback<T>,
    options: Partial<StreamSubscription> = {}
  ): string {
    const subscription: StreamSubscription = {
      id: options.id || `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      callback: callback as StreamCallback<unknown>,
      priority: options.priority || StreamPriority.Medium,
      filters: options.filters
    };

    this.subscriptions.set(subscription.id, subscription);
    return subscription.id;
  }

  /**
   * Unsubscribe from updates
   */
  public unsubscribe(subscriptionId: string): boolean {
    return this.subscriptions.delete(subscriptionId);
  }

  /**
   * Notify subscribers of updates
   */
  private async notifySubscribers(update: RealTimeUpdate): Promise<void> {
    const relevantSubscriptions = Array.from(this.subscriptions.values())
      .filter(sub => this.matchesSubscription(sub, update))
      .sort((a, b) => a.priority - b.priority);

    const metadata: StreamMetadata = {
      timestamp: update.timestamp,
      sequence: update.sequence,
      size: JSON.stringify(update).length,
      compression: 'none',
      latency: Date.now() - update.timestamp
    };

    // Notify subscribers with priority ordering
    for (const subscription of relevantSubscriptions) {
      try {
        const result = subscription.callback(update.data, metadata);
        if (result instanceof Promise) {
          await result;
        }
      } catch (error) {
        console.error(`Error in subscription callback ${subscription.id}:`, error);
      }
    }
  }

  /**
   * Check if update matches subscription filters
   */
  private matchesSubscription(subscription: StreamSubscription, update: RealTimeUpdate): boolean {
    // Type matching
    if (subscription.type !== StreamType.All && subscription.type !== update.type) {
      return false;
    }

    // Apply filters if present
    if (subscription.filters) {
      if (update.type === 'agent_position') {
        const agentUpdate = update.data as AgentPositionUpdate;

        if (subscription.filters.agentIds &&
            !subscription.filters.agentIds.includes(agentUpdate.agentId)) {
          return false;
        }

        if (subscription.filters.agentTypes &&
            !subscription.filters.agentTypes.includes(agentUpdate.agentId.split('_')[1])) {
          return false;
        }
      }

      if (subscription.filters.maxUpdateRate) {
        // Implement rate limiting logic here
        // This would require tracking last update times per subscription
      }
    }

    return true;
  }

  // =============================================================================
  // Data Retrieval
  // =============================================================================

  /**
   * Get current agent positions with interpolation
   */
  public getCurrentAgentPositions(timestamp: number = Date.now()): Map<string, Position3D> {
    const positions = new Map<string, Position3D>();

    this.agentHistories.forEach((history, agentId) => {
      const recent = history.positions.getLatest(2);

      if (recent.length === 0) return;

      if (recent.length === 1) {
        positions.set(agentId, recent[0].position);
        return;
      }

      // Interpolate between two most recent positions
      const interpolated = PositionInterpolator.smoothInterpolate(
        recent[0],
        recent[1],
        timestamp
      );

      positions.set(agentId, interpolated);
    });

    return positions;
  }

  /**
   * Get agent position history
   */
  public getAgentHistory(agentId: string, limit?: number): TimestampedPosition[] {
    const history = this.agentHistories.get(agentId);
    if (!history) return [];

    return limit ? history.positions.getLatest(limit) : history.positions.getAll();
  }

  /**
   * Get recent performance metrics
   */
  public getRecentPerformanceMetrics(limit: number = 100): PerformanceUpdate[] {
    return this.performanceBuffer.getLatest(limit);
  }

  /**
   * Get system status updates
   */
  public getRecentStatusUpdates(limit: number = 50): StatusIndicator[] {
    return this.statusBuffer.getLatest(limit);
  }

  /**
   * Get streaming statistics
   */
  public getStreamingStats(): StreamingStats {
    return {
      connectionState: this.connectionState,
      subscriberCount: this.subscriptions.size,
      agentCount: this.agentHistories.size,
      performanceBufferSize: this.performanceBuffer.size,
      statusBufferSize: this.statusBuffer.size,
      reconnectAttempts: this.reconnectAttempts,
      lastHeartbeat: this.lastHeartbeat,
      sequenceNumber: this.sequenceNumber
    };
  }

  // =============================================================================
  // Utility Methods
  // =============================================================================

  /**
   * Initialize data buffers
   */
  private initializeBuffers(): void {
    // Clear existing buffers
    this.agentHistories.clear();
    this.performanceBuffer.clear();
    this.statusBuffer.clear();
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: Date.now()
        }));
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Start update processing loop
   */
  private startUpdateLoop(): void {
    this.updateTimer = setInterval(() => {
      this.processUpdates();
    }, this.config.updateInterval);
  }

  /**
   * Process pending updates
   */
  private processUpdates(): void {
    // Apply mobile optimizations
    if (this.config.mobileOptimizations.batchedUpdates) {
      this.processBatchedUpdates();
    }

    // Memory cleanup
    this.performMemoryCleanup();
  }

  /**
   * Process batched updates for mobile optimization
   */
  private processBatchedUpdates(): void {
    // Implement batching logic for mobile devices
    // This would group multiple small updates into larger batches
  }

  /**
   * Perform memory cleanup
   */
  private performMemoryCleanup(): void {
    // Clean up old agent histories for disconnected agents
    const cutoffTime = Date.now() - 300000; // 5 minutes

    this.agentHistories.forEach((history, agentId) => {
      const recent = history.positions.getLatest(1);
      if (recent.length === 0 || recent[0].timestamp < cutoffTime) {
        this.agentHistories.delete(agentId);
      }
    });
  }

  /**
   * Stop all timers
   */
  private stopTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
  }

  /**
   * Clean up resources
   */
  public dispose(): void {
    this.disconnect();
    this.subscriptions.clear();
    this.agentHistories.clear();
    this.performanceBuffer.clear();
    this.statusBuffer.clear();
  }
}

// =============================================================================
// Additional Types and Enums
// =============================================================================

/** WebSocket connection states */
export enum ConnectionState {
  Disconnected = 'disconnected',
  Connecting = 'connecting',
  Connected = 'connected',
  Disconnecting = 'disconnecting',
  Error = 'error',
  Reconnecting = 'reconnecting'
}

/** Streaming statistics */
export interface StreamingStats {
  readonly connectionState: ConnectionState;
  readonly subscriberCount: number;
  readonly agentCount: number;
  readonly performanceBufferSize: number;
  readonly statusBufferSize: number;
  readonly reconnectAttempts: number;
  readonly lastHeartbeat: number;
  readonly sequenceNumber: number;
}

// =============================================================================
// Gradio Integration Helpers
// =============================================================================

/**
 * Create Gradio-compatible streaming configuration
 */
export function createGradioStreamingConfig(isMobile: boolean = false): StreamingConfig {
  return {
    websocketUrl: 'ws://localhost:7860/ws', // Will be configured for actual deployment
    updateInterval: isMobile ? 1000 : 500, // Slower updates on mobile
    maxBufferSize: isMobile ? 500 : 1000,
    compressionEnabled: isMobile,
    batchSize: isMobile ? 10 : 5,
    reconnectAttempts: 5,
    heartbeatInterval: 30000, // 30 seconds
    mobileOptimizations: {
      reducedAnimations: isMobile,
      simplifiedPlots: isMobile,
      batchedUpdates: true,
      lowPowerMode: isMobile,
      touchOptimized: isMobile
    }
  };
}

/**
 * Create streaming manager for Gradio interface
 */
export function createStreamingManager(config?: Partial<StreamingConfig>): RealTimeStreamingManager {
  const isMobile = window.innerWidth <= 768;
  const defaultConfig = createGradioStreamingConfig(isMobile);
  const finalConfig = { ...defaultConfig, ...config };

  return new RealTimeStreamingManager(finalConfig);
}

/**
 * Progress tracking utility for Gradio Progress component
 */
export class GradioProgressTracker {
  private progressCallback?: (progress: number, description?: string) => void;

  constructor(progressCallback?: (progress: number, description?: string) => void) {
    this.progressCallback = progressCallback;
  }

  public updateProgress(progress: number, description?: string): void {
    if (this.progressCallback) {
      this.progressCallback(Math.max(0, Math.min(1, progress)), description);
    }
  }

  public setDescription(description: string): void {
    if (this.progressCallback) {
      this.progressCallback(undefined as any, description);
    }
  }

  public complete(): void {
    this.updateProgress(1.0, 'Completed');
  }

  public error(message: string): void {
    this.updateProgress(1.0, `Error: ${message}`);
  }
}

// Export for use in Gradio interface
export default RealTimeStreamingManager;