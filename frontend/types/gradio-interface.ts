/**
 * Felix Framework - Gradio Interface TypeScript Definitions
 *
 * Type-safe definitions for Gradio interface components, event handling,
 * and real-time communication. Provides comprehensive typing for the
 * Felix Framework web interface with performance optimizations.
 *
 * @version 1.0.0
 * @author Felix Framework Team
 */

import {
  AgentType,
  AgentState,
  Position3D,
  HelixParameter,
  FelixTask,
  TaskResult,
  PerformanceMetrics,
  FelixSession,
  ValidationResult
} from './felix-core';

// =============================================================================
// Gradio Component Types
// =============================================================================

/** Gradio component base interface */
export interface GradioComponent<T = unknown> {
  readonly value: T;
  readonly label?: string;
  readonly info?: string;
  readonly visible?: boolean;
  readonly interactive?: boolean;
  readonly placeholder?: string;
  readonly className?: string;
}

/** Gradio input component types */
export interface GradioTextbox extends GradioComponent<string> {
  readonly lines?: number;
  readonly maxLines?: number;
  readonly type?: 'text' | 'password' | 'email';
  readonly autofocus?: boolean;
  readonly showCopyButton?: boolean;
}

export interface GradioSlider extends GradioComponent<number> {
  readonly minimum: number;
  readonly maximum: number;
  readonly step?: number;
  readonly randomize?: boolean;
  readonly showLabel?: boolean;
}

export interface GradioCheckboxGroup extends GradioComponent<readonly string[]> {
  readonly choices: readonly string[];
  readonly type?: 'value' | 'index';
  readonly interactive?: boolean;
}

export interface GradioCheckbox extends GradioComponent<boolean> {
  readonly container?: boolean;
  readonly showLabel?: boolean;
}

export interface GradioButton extends GradioComponent<null> {
  readonly variant?: 'primary' | 'secondary' | 'stop';
  readonly size?: 'sm' | 'lg';
  readonly icon?: string;
  readonly link?: string;
}

export interface GradioDropdown extends GradioComponent<string | string[]> {
  readonly choices: readonly string[];
  readonly multiselect?: boolean;
  readonly allowCustomValue?: boolean;
  readonly filterable?: boolean;
}

// =============================================================================
// Gradio Output Component Types
// =============================================================================

/** Plotly.js figure configuration for Gradio */
export interface GradioPlot extends GradioComponent<PlotlyFigure> {
  readonly height?: number;
  readonly width?: number;
}

/** Plotly figure structure */
export interface PlotlyFigure {
  readonly data: readonly PlotlyTrace[];
  readonly layout: PlotlyLayout;
  readonly config?: PlotlyConfig;
}

/** Plotly trace data */
export interface PlotlyTrace {
  readonly type: PlotlyTraceType;
  readonly x?: readonly number[];
  readonly y?: readonly number[];
  readonly z?: readonly number[];
  readonly mode?: string;
  readonly name?: string;
  readonly marker?: PlotlyMarker;
  readonly line?: PlotlyLine;
  readonly hovertemplate?: string;
  readonly showlegend?: boolean;
  readonly [key: string]: unknown;
}

/** Plotly trace types */
export type PlotlyTraceType =
  | 'scatter3d'
  | 'scatter'
  | 'bar'
  | 'pie'
  | 'line'
  | 'surface'
  | 'mesh3d';

/** Plotly marker configuration */
export interface PlotlyMarker {
  readonly color?: string | readonly string[];
  readonly size?: number | readonly number[];
  readonly opacity?: number;
  readonly symbol?: string;
  readonly line?: PlotlyLine;
  readonly colorscale?: string;
  readonly showscale?: boolean;
}

/** Plotly line configuration */
export interface PlotlyLine {
  readonly color?: string;
  readonly width?: number;
  readonly dash?: string;
}

/** Plotly layout configuration */
export interface PlotlyLayout {
  readonly title?: PlotlyTitle;
  readonly scene?: PlotlyScene;
  readonly xaxis?: PlotlyAxis;
  readonly yaxis?: PlotlyAxis;
  readonly zaxis?: PlotlyAxis;
  readonly width?: number;
  readonly height?: number;
  readonly margin?: PlotlyMargin;
  readonly bgcolor?: string;
  readonly paper_bgcolor?: string;
  readonly font?: PlotlyFont;
  readonly showlegend?: boolean;
  readonly legend?: PlotlyLegend;
  readonly annotations?: readonly PlotlyAnnotation[];
}

/** Plotly title configuration */
export interface PlotlyTitle {
  readonly text: string;
  readonly x?: number;
  readonly y?: number;
  readonly font?: PlotlyFont;
}

/** Plotly 3D scene configuration */
export interface PlotlyScene {
  readonly xaxis?: PlotlyAxis;
  readonly yaxis?: PlotlyAxis;
  readonly zaxis?: PlotlyAxis;
  readonly camera?: PlotlyCamera;
  readonly aspectmode?: 'auto' | 'cube' | 'data' | 'manual';
  readonly aspectratio?: { x: number; y: number; z: number };
  readonly bgcolor?: string;
}

/** Plotly axis configuration */
export interface PlotlyAxis {
  readonly title?: string | PlotlyTitle;
  readonly range?: readonly [number, number];
  readonly type?: 'linear' | 'log' | 'date' | 'category';
  readonly showgrid?: boolean;
  readonly gridcolor?: string;
  readonly showticklabels?: boolean;
  readonly tickformat?: string;
}

/** Plotly camera configuration */
export interface PlotlyCamera {
  readonly up?: { x: number; y: number; z: number };
  readonly center?: { x: number; y: number; z: number };
  readonly eye?: { x: number; y: number; z: number };
  readonly projection?: { type: 'perspective' | 'orthographic' };
}

/** Plotly margin configuration */
export interface PlotlyMargin {
  readonly l?: number;
  readonly r?: number;
  readonly t?: number;
  readonly b?: number;
}

/** Plotly font configuration */
export interface PlotlyFont {
  readonly family?: string;
  readonly size?: number;
  readonly color?: string;
}

/** Plotly legend configuration */
export interface PlotlyLegend {
  readonly x?: number;
  readonly y?: number;
  readonly bgcolor?: string;
  readonly bordercolor?: string;
  readonly borderwidth?: number;
}

/** Plotly annotation */
export interface PlotlyAnnotation {
  readonly text: string;
  readonly x: number;
  readonly y: number;
  readonly z?: number;
  readonly font?: PlotlyFont;
  readonly bgcolor?: string;
  readonly bordercolor?: string;
  readonly arrow?: PlotlyArrow;
}

/** Plotly arrow configuration */
export interface PlotlyArrow {
  readonly head?: number;
  readonly size?: number;
  readonly width?: number;
  readonly color?: string;
}

/** Plotly configuration options */
export interface PlotlyConfig {
  readonly displayModeBar?: boolean;
  readonly responsive?: boolean;
  readonly doubleClick?: 'reset' | 'autosize' | false;
  readonly showTips?: boolean;
  readonly showLink?: boolean;
  readonly linkText?: string;
  readonly toImageButtonOptions?: PlotlyImageOptions;
}

/** Plotly image export options */
export interface PlotlyImageOptions {
  readonly format?: 'png' | 'jpeg' | 'webp' | 'svg' | 'html';
  readonly width?: number;
  readonly height?: number;
  readonly scale?: number;
  readonly filename?: string;
}

/** Gradio JSON output component */
export interface GradioJSON extends GradioComponent<Record<string, unknown> | readonly unknown[]> {
  readonly open?: boolean;
}

/** Gradio Markdown output component */
export interface GradioMarkdown extends GradioComponent<string> {
  readonly height?: number;
  readonly linkify?: boolean;
  readonly sanitizeHtml?: boolean;
  readonly latexDelimiters?: readonly { left: string; right: string; display: boolean }[];
}

/** Gradio HTML output component */
export interface GradioHTML extends GradioComponent<string> {
  readonly showLabel?: boolean;
}

// =============================================================================
// Felix-Specific Gradio Interface Types
// =============================================================================

/** Felix task input interface */
export interface FelixTaskInput {
  readonly description: GradioTextbox;
  readonly agentTypes: GradioCheckboxGroup;
  readonly maxAgents: GradioSlider;
  readonly priority: GradioDropdown;
  readonly timeout: GradioSlider;
  readonly enableLLM: GradioCheckbox;
}

/** Felix visualization interface */
export interface FelixVisualization {
  readonly helixPlot: GradioPlot;
  readonly performancePlot: GradioPlot;
  readonly showAgents: GradioCheckbox;
  readonly highlightActive: GradioCheckbox;
  readonly agentFilter: GradioCheckboxGroup;
  readonly animationSpeed: GradioSlider;
  readonly viewMode: GradioDropdown;
}

/** Felix output interface */
export interface FelixOutput {
  readonly resultText: GradioMarkdown;
  readonly performanceMetrics: GradioJSON;
  readonly sessionStatus: GradioJSON;
  readonly errorLog: GradioMarkdown;
  readonly exportData: GradioJSON;
}

/** Felix control interface */
export interface FelixControls {
  readonly processButton: GradioButton;
  readonly stopButton: GradioButton;
  readonly clearButton: GradioButton;
  readonly exportButton: GradioButton;
  readonly newSessionButton: GradioButton;
  readonly debugToggle: GradioCheckbox;
}

// =============================================================================
// Event Handling Types
// =============================================================================

/** Gradio event types */
export enum GradioEventType {
  Click = 'click',
  Change = 'change',
  Submit = 'submit',
  Upload = 'upload',
  Clear = 'clear',
  Focus = 'focus',
  Blur = 'blur',
  KeyPress = 'keypress'
}

/** Gradio event handler function signature */
export type GradioEventHandler<T = unknown> = (
  value: T,
  ...additionalArgs: readonly unknown[]
) => Promise<GradioEventResult> | GradioEventResult;

/** Gradio event result */
export interface GradioEventResult {
  readonly updates?: Record<string, GradioComponentUpdate>;
  readonly outputs?: readonly unknown[];
  readonly errors?: readonly string[];
  readonly success: boolean;
}

/** Gradio component update */
export interface GradioComponentUpdate {
  readonly value?: unknown;
  readonly visible?: boolean;
  readonly interactive?: boolean;
  readonly label?: string;
  readonly choices?: readonly string[];
}

/** Felix-specific event types */
export enum FelixEventType {
  TaskStarted = 'task_started',
  TaskCompleted = 'task_completed',
  TaskFailed = 'task_failed',
  AgentSpawned = 'agent_spawned',
  AgentCompleted = 'agent_completed',
  VisualizationUpdated = 'visualization_updated',
  PerformanceUpdated = 'performance_updated',
  SessionStateChanged = 'session_state_changed',
  ErrorOccurred = 'error_occurred'
}

/** Felix event data */
export interface FelixEventData<T = unknown> {
  readonly type: FelixEventType;
  readonly timestamp: number;
  readonly sessionId: string;
  readonly payload: T;
  readonly source: 'user' | 'system' | 'agent';
}

/** Felix event handlers */
export interface FelixEventHandlers {
  readonly onTaskStarted: GradioEventHandler<FelixTask>;
  readonly onTaskCompleted: GradioEventHandler<TaskResult>;
  readonly onTaskFailed: GradioEventHandler<{ taskId: string; error: string }>;
  readonly onAgentSpawned: GradioEventHandler<{ agentId: string; type: AgentType; position: Position3D }>;
  readonly onAgentCompleted: GradioEventHandler<{ agentId: string; result: TaskResult }>;
  readonly onVisualizationUpdated: GradioEventHandler<PlotlyFigure>;
  readonly onPerformanceUpdated: GradioEventHandler<PerformanceMetrics>;
  readonly onSessionStateChanged: GradioEventHandler<FelixSession>;
  readonly onErrorOccurred: GradioEventHandler<{ error: string; context?: Record<string, unknown> }>;
}

// =============================================================================
// Real-time Update Types
// =============================================================================

/** Real-time update configuration */
export interface RealTimeConfig {
  readonly enabled: boolean;
  readonly updateInterval: number; // milliseconds
  readonly maxUpdateRate: number; // updates per second
  readonly bufferSize: number;
  readonly compression: 'none' | 'gzip' | 'lz4';
  readonly batchUpdates: boolean;
}

/** Real-time update message */
export interface RealTimeUpdate<T = unknown> {
  readonly id: string;
  readonly type: string;
  readonly timestamp: number;
  readonly sequence: number;
  readonly data: T;
  readonly metadata?: Record<string, unknown>;
}

/** Agent position update */
export interface AgentPositionUpdate {
  readonly agentId: string;
  readonly position: Position3D;
  readonly helixParameter: HelixParameter;
  readonly state: AgentState;
  readonly timestamp: number;
}

/** Performance metrics update */
export interface PerformanceUpdate {
  readonly metrics: PerformanceMetrics;
  readonly timestamp: number;
  readonly delta?: Partial<PerformanceMetrics>;
}

/** Visualization update batch */
export interface VisualizationUpdateBatch {
  readonly agentUpdates: readonly AgentPositionUpdate[];
  readonly performanceUpdate?: PerformanceUpdate;
  readonly plotUpdate?: Partial<PlotlyFigure>;
  readonly timestamp: number;
}

// =============================================================================
// Progress and Status Types
// =============================================================================

/** Progress tracking interface */
export interface ProgressTracker {
  readonly total: number;
  readonly completed: number;
  readonly percentage: number;
  readonly status: ProgressStatus;
  readonly message?: string;
  readonly eta?: number; // estimated time remaining in seconds
}

/** Progress status types */
export enum ProgressStatus {
  NotStarted = 'not_started',
  InProgress = 'in_progress',
  Completed = 'completed',
  Failed = 'failed',
  Cancelled = 'cancelled'
}

/** Status indicator interface */
export interface StatusIndicator {
  readonly type: StatusType;
  readonly message: string;
  readonly timestamp: number;
  readonly details?: Record<string, unknown>;
}

/** Status types */
export enum StatusType {
  Info = 'info',
  Success = 'success',
  Warning = 'warning',
  Error = 'error',
  Loading = 'loading'
}

// =============================================================================
// Mobile and Responsive Design Types
// =============================================================================

/** Responsive layout configuration */
export interface ResponsiveConfig {
  readonly breakpoints: {
    readonly mobile: number;
    readonly tablet: number;
    readonly desktop: number;
  };
  readonly layoutMode: 'adaptive' | 'responsive';
  readonly mobileOptimizations: MobileOptimizations;
}

/** Mobile optimization settings */
export interface MobileOptimizations {
  readonly reducedAnimations: boolean;
  readonly simplifiedPlots: boolean;
  readonly batchedUpdates: boolean;
  readonly lowPowerMode: boolean;
  readonly touchOptimized: boolean;
}

/** Viewport configuration */
export interface ViewportConfig {
  readonly width: number;
  readonly height: number;
  readonly devicePixelRatio: number;
  readonly isMobile: boolean;
  readonly isTablet: boolean;
  readonly orientation: 'portrait' | 'landscape';
}

// =============================================================================
// Educational Content Types
// =============================================================================

/** Educational content section */
export interface EducationalContent {
  readonly id: string;
  readonly title: string;
  readonly content: string;
  readonly type: ContentType;
  readonly difficulty: DifficultyLevel;
  readonly estimatedTime: number; // minutes
  readonly prerequisites: readonly string[];
  readonly resources: readonly ContentResource[];
}

/** Content types */
export enum ContentType {
  Introduction = 'introduction',
  Tutorial = 'tutorial',
  Example = 'example',
  Reference = 'reference',
  Research = 'research'
}

/** Difficulty levels */
export enum DifficultyLevel {
  Beginner = 'beginner',
  Intermediate = 'intermediate',
  Advanced = 'advanced',
  Expert = 'expert'
}

/** Content resource */
export interface ContentResource {
  readonly type: 'link' | 'file' | 'video' | 'paper';
  readonly title: string;
  readonly url: string;
  readonly description?: string;
}

/** Guided tour step */
export interface TourStep {
  readonly id: string;
  readonly target: string; // CSS selector or component ID
  readonly title: string;
  readonly content: string;
  readonly position: 'top' | 'bottom' | 'left' | 'right' | 'center';
  readonly action?: TourAction;
  readonly validation?: TourValidation;
}

/** Tour action */
export interface TourAction {
  readonly type: 'click' | 'input' | 'wait' | 'highlight';
  readonly value?: string;
  readonly duration?: number;
}

/** Tour validation */
export interface TourValidation {
  readonly condition: (state: unknown) => boolean;
  readonly message: string;
  readonly retry: boolean;
}

// =============================================================================
// Export Configuration Types
// =============================================================================

/** Export format options */
export enum ExportFormat {
  JSON = 'json',
  CSV = 'csv',
  PNG = 'png',
  SVG = 'svg',
  PDF = 'pdf',
  HTML = 'html'
}

/** Export configuration */
export interface ExportConfig {
  readonly format: ExportFormat;
  readonly includeVisualization: boolean;
  readonly includeMetrics: boolean;
  readonly includeAgentData: boolean;
  readonly includeSessionData: boolean;
  readonly compression: boolean;
  readonly filename?: string;
}

/** Export result */
export interface ExportResult {
  readonly success: boolean;
  readonly data?: string | ArrayBuffer;
  readonly filename: string;
  readonly size: number;
  readonly format: ExportFormat;
  readonly error?: string;
}

// =============================================================================
// Type Guards and Utilities
// =============================================================================

/** Type guards for Gradio interface validation */
export namespace GradioTypeGuards {
  export function isPlotlyFigure(value: unknown): value is PlotlyFigure {
    return typeof value === 'object' && value !== null &&
           'data' in value && 'layout' in value &&
           Array.isArray((value as any).data);
  }

  export function isRealTimeUpdate(value: unknown): value is RealTimeUpdate {
    return typeof value === 'object' && value !== null &&
           'id' in value && 'type' in value && 'timestamp' in value &&
           typeof (value as any).id === 'string' &&
           typeof (value as any).timestamp === 'number';
  }

  export function isValidationResult(value: unknown): value is ValidationResult {
    return typeof value === 'object' && value !== null &&
           'valid' in value && typeof (value as any).valid === 'boolean';
  }
}

// =============================================================================
// Interface Configuration
// =============================================================================

/** Complete Felix Gradio interface configuration */
export interface FelixGradioConfig {
  readonly title: string;
  readonly description: string;
  readonly theme: 'light' | 'dark' | 'auto';
  readonly allowFlagging: boolean;
  readonly showError: boolean;
  readonly enableQueue: boolean;
  readonly maxThreads: number;
  readonly auth?: (username: string, password: string) => boolean;
  readonly authMessage?: string;
  readonly css?: string;
  readonly js?: string;
  readonly favicon?: string;
  readonly realTime: RealTimeConfig;
  readonly responsive: ResponsiveConfig;
  readonly analytics: AnalyticsConfig;
}

/** Analytics configuration */
export interface AnalyticsConfig {
  readonly enabled: boolean;
  readonly trackEvents: readonly string[];
  readonly trackPerformance: boolean;
  readonly trackErrors: boolean;
  readonly endpoint?: string;
  readonly apiKey?: string;
}

// Export all types for external use
export type {
  // Component types
  GradioComponent,
  GradioTextbox,
  GradioSlider,
  GradioCheckboxGroup,
  GradioCheckbox,
  GradioButton,
  GradioDropdown,
  GradioPlot,
  GradioJSON,
  GradioMarkdown,
  GradioHTML,

  // Plotly types
  PlotlyFigure,
  PlotlyTrace,
  PlotlyLayout,
  PlotlyMarker,
  PlotlyLine,
  PlotlyConfig,

  // Felix interface types
  FelixTaskInput,
  FelixVisualization,
  FelixOutput,
  FelixControls,

  // Event types
  GradioEventHandler,
  GradioEventResult,
  FelixEventData,
  FelixEventHandlers,

  // Real-time types
  RealTimeConfig,
  RealTimeUpdate,
  AgentPositionUpdate,
  PerformanceUpdate,
  VisualizationUpdateBatch,

  // Progress types
  ProgressTracker,
  StatusIndicator,

  // Responsive types
  ResponsiveConfig,
  MobileOptimizations,
  ViewportConfig,

  // Educational types
  EducationalContent,
  TourStep,

  // Export types
  ExportConfig,
  ExportResult,

  // Configuration types
  FelixGradioConfig,
  AnalyticsConfig
};

export {
  // Enums
  GradioEventType,
  FelixEventType,
  PlotlyTraceType,
  ProgressStatus,
  StatusType,
  ContentType,
  DifficultyLevel,
  ExportFormat,

  // Type guards
  GradioTypeGuards
};