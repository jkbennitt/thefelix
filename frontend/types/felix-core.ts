/**
 * Felix Framework - Core TypeScript Definitions
 *
 * Type-safe definitions for Felix Framework's helix-based multi-agent
 * cognitive architecture. Ensures type safety across frontend-backend
 * communication and provides comprehensive mathematical typing.
 *
 * @version 1.0.0
 * @author Felix Framework Team
 */

// =============================================================================
// Mathematical and Geometric Types
// =============================================================================

/** Branded type for mathematical precision */
export type HelixParameter = number & { readonly __brand: 'HelixParameter' };
export type Coordinate3D = number & { readonly __brand: 'Coordinate3D' };
export type Radius = number & { readonly __brand: 'Radius' };
export type Turns = number & { readonly __brand: 'Turns' };
export type Height = number & { readonly __brand: 'Height' };

/** 3D position vector with type safety */
export interface Position3D {
  readonly x: Coordinate3D;
  readonly y: Coordinate3D;
  readonly z: Coordinate3D;
}

/** Helix geometry configuration */
export interface HelixGeometry {
  readonly topRadius: Radius;
  readonly bottomRadius: Radius;
  readonly height: Height;
  readonly turns: Turns;
  readonly concentrationRatio: number;
  readonly mathematicalPrecision: number;
}

/** Parametric helix point with validation */
export interface HelixPoint {
  readonly t: HelixParameter; // t ∈ [0,1]
  readonly position: Position3D;
  readonly radius: Radius;
  readonly angle: number; // θ(t) = 2πnt
  readonly height: Height;
  readonly timestamp: number;
}

/** Helix path segment for efficient rendering */
export interface HelixSegment {
  readonly startT: HelixParameter;
  readonly endT: HelixParameter;
  readonly points: readonly HelixPoint[];
  readonly resolution: number;
  readonly boundingBox: {
    readonly min: Position3D;
    readonly max: Position3D;
  };
}

// =============================================================================
// Agent System Types
// =============================================================================

/** Agent types with specialized capabilities */
export enum AgentType {
  Research = 'research',
  Analysis = 'analysis',
  Synthesis = 'synthesis',
  Critic = 'critic',
  General = 'general'
}

/** Agent lifecycle states */
export enum AgentState {
  Spawning = 'spawning',
  Active = 'active',
  Processing = 'processing',
  Waiting = 'waiting',
  Completed = 'completed',
  Failed = 'failed',
  Terminated = 'terminated'
}

/** Agent configuration with helix-specific properties */
export interface AgentConfig {
  readonly id: string;
  readonly type: AgentType;
  readonly spawnTime: number;
  readonly helixPosition: HelixParameter;
  readonly maxLifetime: number;
  readonly temperature: number; // LLM creativity parameter
  readonly capabilities: readonly string[];
  readonly modelPreference?: string;
}

/** Agent runtime state */
export interface AgentInstance {
  readonly config: AgentConfig;
  readonly state: AgentState;
  readonly currentPosition: Position3D;
  readonly spawnTimestamp: number;
  readonly lastActivity: number;
  readonly processedTasks: number;
  readonly performance: AgentPerformance;
  readonly communication: AgentCommunication;
}

/** Agent performance metrics */
export interface AgentPerformance {
  readonly responseTime: number;
  readonly successRate: number;
  readonly tokenUsage: number;
  readonly errorCount: number;
  readonly qualityScore: number;
  readonly efficiencyRating: number;
}

/** Agent communication state */
export interface AgentCommunication {
  readonly messagesSent: number;
  readonly messagesReceived: number;
  readonly spokeConnections: readonly string[];
  readonly lastSpokeActivity: number;
  readonly communicationLoad: number;
}

// =============================================================================
// Communication System Types
// =============================================================================

/** Message types in Felix communication system */
export enum MessageType {
  TaskRequest = 'task_request',
  TaskResponse = 'task_response',
  StatusUpdate = 'status_update',
  CoordinationSignal = 'coordination_signal',
  ErrorReport = 'error_report',
  PerformanceMetric = 'performance_metric',
  SystemControl = 'system_control'
}

/** Message priority levels */
export enum MessagePriority {
  Critical = 0,
  High = 1,
  Medium = 2,
  Low = 3,
  Background = 4
}

/** Felix Framework message structure */
export interface FelixMessage {
  readonly id: string;
  readonly type: MessageType;
  readonly priority: MessagePriority;
  readonly senderId: string;
  readonly receiverId: string | null; // null for broadcast
  readonly timestamp: number;
  readonly ttl: number; // time to live
  readonly payload: unknown;
  readonly metadata: MessageMetadata;
}

/** Message metadata for routing and analytics */
export interface MessageMetadata {
  readonly helixPosition: HelixParameter;
  readonly spokeId: string;
  readonly retryCount: number;
  readonly processingTime?: number;
  readonly responseRequired: boolean;
  readonly correlationId?: string;
}

/** Central Post coordination state */
export interface CentralPostState {
  readonly activeAgents: number;
  readonly messageQueue: number;
  readonly systemLoad: number;
  readonly coordinationEfficiency: number;
  readonly spokeUtilization: Record<string, number>;
  readonly lastUpdate: number;
}

// =============================================================================
// Task Processing Types
// =============================================================================

/** Task definition with Felix-specific properties */
export interface FelixTask {
  readonly id: string;
  readonly description: string;
  readonly type: TaskType;
  readonly priority: TaskPriority;
  readonly requirements: TaskRequirements;
  readonly constraints: TaskConstraints;
  readonly expectedOutput: TaskOutputSpec;
  readonly createdAt: number;
  readonly deadline?: number;
}

/** Task processing types */
export enum TaskType {
  Research = 'research',
  Analysis = 'analysis',
  Synthesis = 'synthesis',
  Review = 'review',
  Creative = 'creative',
  Technical = 'technical',
  Collaborative = 'collaborative'
}

/** Task priority levels */
export enum TaskPriority {
  Urgent = 0,
  High = 1,
  Medium = 2,
  Low = 3,
  Background = 4
}

/** Task requirements specification */
export interface TaskRequirements {
  readonly agentTypes: readonly AgentType[];
  readonly minAgents: number;
  readonly maxAgents: number;
  readonly requiredCapabilities: readonly string[];
  readonly estimatedTokens: number;
  readonly timeoutSeconds: number;
}

/** Task processing constraints */
export interface TaskConstraints {
  readonly maxTokenBudget: number;
  readonly allowParallel: boolean;
  readonly requireConsensus: boolean;
  readonly qualityThreshold: number;
  readonly retryLimit: number;
}

/** Expected output specification */
export interface TaskOutputSpec {
  readonly format: 'text' | 'json' | 'markdown' | 'structured';
  readonly minLength?: number;
  readonly maxLength?: number;
  readonly requiredSections?: readonly string[];
  readonly validationRules?: readonly ValidationRule[];
}

/** Validation rules for output quality */
export interface ValidationRule {
  readonly name: string;
  readonly type: 'format' | 'content' | 'length' | 'custom';
  readonly rule: string | RegExp | ((value: unknown) => boolean);
  readonly errorMessage: string;
}

/** Task execution result */
export interface TaskResult {
  readonly taskId: string;
  readonly success: boolean;
  readonly output: unknown;
  readonly agentsUsed: readonly string[];
  readonly processingTime: number;
  readonly tokenUsage: number;
  readonly qualityScore: number;
  readonly errors: readonly TaskError[];
  readonly metadata: TaskResultMetadata;
}

/** Task processing error */
export interface TaskError {
  readonly code: string;
  readonly message: string;
  readonly agentId?: string;
  readonly timestamp: number;
  readonly severity: 'warning' | 'error' | 'critical';
  readonly context?: Record<string, unknown>;
}

/** Task result metadata */
export interface TaskResultMetadata {
  readonly helixCoordination: HelixCoordinationStats;
  readonly communicationStats: CommunicationStats;
  readonly performanceMetrics: PerformanceMetrics;
  readonly convergencePattern: ConvergencePattern;
}

// =============================================================================
// Performance and Analytics Types
// =============================================================================

/** System-wide performance metrics */
export interface PerformanceMetrics {
  readonly throughput: number; // tasks per minute
  readonly latency: number; // average response time
  readonly errorRate: number; // percentage of failed tasks
  readonly resourceUtilization: ResourceUtilization;
  readonly scalabilityMetrics: ScalabilityMetrics;
  readonly timestamp: number;
}

/** Resource utilization tracking */
export interface ResourceUtilization {
  readonly memory: MemoryUsage;
  readonly cpu: number; // percentage
  readonly network: NetworkUsage;
  readonly storage: number; // bytes used
  readonly tokens: TokenUsage;
}

/** Memory usage breakdown */
export interface MemoryUsage {
  readonly total: number;
  readonly used: number;
  readonly helix: number;
  readonly agents: number;
  readonly messages: number;
  readonly cache: number;
}

/** Network communication statistics */
export interface NetworkUsage {
  readonly bytesIn: number;
  readonly bytesOut: number;
  readonly requestsPerSecond: number;
  readonly activeConnections: number;
  readonly bandwidth: number;
}

/** Token usage for LLM operations */
export interface TokenUsage {
  readonly total: number;
  readonly used: number;
  readonly remaining: number;
  readonly costEstimate: number;
  readonly byAgentType: Record<AgentType, number>;
}

/** Scalability performance metrics */
export interface ScalabilityMetrics {
  readonly maxConcurrentAgents: number;
  readonly communicationComplexity: 'O(N)' | 'O(N²)' | 'O(log N)';
  readonly memoryComplexity: 'O(N)' | 'O(N²)' | 'O(log N)';
  readonly bottlenecks: readonly string[];
  readonly scalabilityScore: number;
}

/** Helix-specific coordination statistics */
export interface HelixCoordinationStats {
  readonly agentDistribution: Record<HelixParameter, number>;
  readonly convergenceTime: number;
  readonly focusingEfficiency: number;
  readonly spokeUtilization: number;
  readonly geometricAdvantage: number;
}

/** Communication pattern analysis */
export interface CommunicationStats {
  readonly messageVolume: number;
  readonly averageLatency: number;
  readonly spokeEfficiency: number;
  readonly congestionPoints: readonly string[];
  readonly patternAnalysis: CommunicationPattern;
}

/** Communication pattern types */
export interface CommunicationPattern {
  readonly type: 'spoke' | 'mesh' | 'hybrid';
  readonly efficiency: number;
  readonly complexity: string;
  readonly advantages: readonly string[];
  readonly limitations: readonly string[];
}

/** Convergence pattern analysis */
export interface ConvergencePattern {
  readonly type: 'geometric' | 'linear' | 'exponential';
  readonly rate: number;
  readonly stability: number;
  readonly predictability: number;
  readonly quality: number;
}

// =============================================================================
// Session and State Management Types
// =============================================================================

/** Felix Framework session state */
export interface FelixSession {
  readonly sessionId: string;
  readonly startTime: number;
  readonly lastActivity: number;
  readonly configuration: SessionConfiguration;
  readonly state: SessionState;
  readonly statistics: SessionStatistics;
  readonly participants: readonly AgentInstance[];
}

/** Session configuration options */
export interface SessionConfiguration {
  readonly maxAgents: number;
  readonly tokenBudget: number;
  readonly timeoutSeconds: number;
  readonly enableLLM: boolean;
  readonly helixConfig: HelixGeometry;
  readonly debugMode: boolean;
  readonly persistState: boolean;
}

/** Session lifecycle states */
export enum SessionState {
  Initializing = 'initializing',
  Active = 'active',
  Processing = 'processing',
  Paused = 'paused',
  Completed = 'completed',
  Error = 'error',
  Terminated = 'terminated'
}

/** Session performance statistics */
export interface SessionStatistics {
  readonly tasksCompleted: number;
  readonly agentsSpawned: number;
  readonly messagesExchanged: number;
  readonly tokensUsed: number;
  readonly averageResponseTime: number;
  readonly successRate: number;
  readonly uptime: number;
}

// =============================================================================
// LLM Integration Types
// =============================================================================

/** LLM model configuration */
export interface LLMModelConfig {
  readonly modelId: string;
  readonly provider: 'huggingface' | 'openai' | 'anthropic' | 'local';
  readonly capabilities: readonly ModelCapability[];
  readonly parameters: ModelParameters;
  readonly limits: ModelLimits;
  readonly pricing: ModelPricing;
}

/** Model capabilities */
export enum ModelCapability {
  TextGeneration = 'text_generation',
  CodeGeneration = 'code_generation',
  Analysis = 'analysis',
  Translation = 'translation',
  Summarization = 'summarization',
  QuestionAnswering = 'question_answering',
  Creative = 'creative'
}

/** Model parameter configuration */
export interface ModelParameters {
  readonly temperature: number;
  readonly maxTokens: number;
  readonly topP: number;
  readonly topK: number;
  readonly repetitionPenalty: number;
  readonly stopSequences: readonly string[];
}

/** Model usage limits */
export interface ModelLimits {
  readonly maxRequestsPerMinute: number;
  readonly maxTokensPerRequest: number;
  readonly maxConcurrentRequests: number;
  readonly dailyTokenLimit: number;
}

/** Model pricing information */
export interface ModelPricing {
  readonly inputTokenCost: number;
  readonly outputTokenCost: number;
  readonly currency: string;
  readonly billingUnit: string;
}

// =============================================================================
// Utility Types and Type Guards
// =============================================================================

/** Type predicate functions for runtime type checking */
export namespace TypeGuards {
  export function isHelixParameter(value: unknown): value is HelixParameter {
    return typeof value === 'number' && value >= 0 && value <= 1;
  }

  export function isPosition3D(value: unknown): value is Position3D {
    return typeof value === 'object' && value !== null &&
           'x' in value && 'y' in value && 'z' in value &&
           typeof (value as any).x === 'number' &&
           typeof (value as any).y === 'number' &&
           typeof (value as any).z === 'number';
  }

  export function isAgentType(value: unknown): value is AgentType {
    return typeof value === 'string' && Object.values(AgentType).includes(value as AgentType);
  }

  export function isAgentState(value: unknown): value is AgentState {
    return typeof value === 'string' && Object.values(AgentState).includes(value as AgentState);
  }

  export function isFelixMessage(value: unknown): value is FelixMessage {
    return typeof value === 'object' && value !== null &&
           'id' in value && 'type' in value && 'senderId' in value &&
           typeof (value as any).id === 'string' &&
           typeof (value as any).senderId === 'string';
  }
}

/** Utility types for type manipulation */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

export type Partial<T> = {
  [P in keyof T]?: T[P];
};

export type Required<T> = {
  [P in keyof T]-?: T[P];
};

/** Event types for reactive updates */
export interface FelixEvent<T = unknown> {
  readonly type: string;
  readonly timestamp: number;
  readonly payload: T;
  readonly source: string;
}

/** Configuration validation result */
export interface ValidationResult {
  readonly valid: boolean;
  readonly errors: readonly ValidationError[];
  readonly warnings: readonly ValidationWarning[];
}

export interface ValidationError {
  readonly field: string;
  readonly message: string;
  readonly code: string;
}

export interface ValidationWarning {
  readonly field: string;
  readonly message: string;
  readonly suggestion?: string;
}

// =============================================================================
// Export all types for external use
// =============================================================================

export type {
  // Core geometric types
  HelixParameter,
  Coordinate3D,
  Radius,
  Turns,
  Height,
  Position3D,
  HelixGeometry,
  HelixPoint,
  HelixSegment,

  // Agent system types
  AgentConfig,
  AgentInstance,
  AgentPerformance,
  AgentCommunication,

  // Communication types
  FelixMessage,
  MessageMetadata,
  CentralPostState,

  // Task processing types
  FelixTask,
  TaskRequirements,
  TaskConstraints,
  TaskOutputSpec,
  ValidationRule,
  TaskResult,
  TaskError,
  TaskResultMetadata,

  // Performance types
  PerformanceMetrics,
  ResourceUtilization,
  MemoryUsage,
  NetworkUsage,
  TokenUsage,
  ScalabilityMetrics,
  HelixCoordinationStats,
  CommunicationStats,
  CommunicationPattern,
  ConvergencePattern,

  // Session types
  FelixSession,
  SessionConfiguration,
  SessionStatistics,

  // LLM types
  LLMModelConfig,
  ModelParameters,
  ModelLimits,
  ModelPricing,

  // Utility types
  DeepReadonly,
  FelixEvent,
  ValidationResult,
  ValidationError,
  ValidationWarning
};

export {
  // Enums
  AgentType,
  AgentState,
  MessageType,
  MessagePriority,
  TaskType,
  TaskPriority,
  SessionState,
  ModelCapability,

  // Type guards
  TypeGuards
};