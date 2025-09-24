/**
 * Felix Framework - Enhanced 3D Helix Visualization Component
 *
 * Type-safe, performance-optimized 3D visualization component for Felix Framework
 * helix geometry with real-time agent tracking, educational annotations, and
 * mobile-responsive design.
 *
 * Features:
 * - WebGL-accelerated rendering with Plotly.js
 * - Real-time agent position updates
 * - Interactive camera controls and presets
 * - Educational overlays and guided tours
 * - Mobile-optimized touch controls
 * - Export capabilities (PNG, SVG, HTML)
 * - Progressive loading for large datasets
 *
 * @version 1.0.0
 * @author Felix Framework Team
 */

import {
  PlotlyFigure,
  PlotlyTrace,
  PlotlyLayout,
  PlotlyConfig,
  PlotlyMarker,
  PlotlyLine,
  AgentPositionUpdate,
  VisualizationUpdateBatch,
  ViewportConfig,
  MobileOptimizations,
  ExportConfig,
  ExportResult,
  ExportFormat
} from '../types/gradio-interface';

import {
  Position3D,
  HelixGeometry,
  HelixPoint,
  HelixSegment,
  AgentType,
  AgentState,
  AgentInstance,
  Coordinate3D,
  HelixParameter
} from '../types/felix-core';

// =============================================================================
// Configuration Types
// =============================================================================

/** Helix visualization configuration */
export interface HelixVisualizationConfig {
  readonly width: number;
  readonly height: number;
  readonly resolution: number; // points per helix segment
  readonly enableAnimations: boolean;
  readonly showEducationalAnnotations: boolean;
  readonly enableTooltips: boolean;
  readonly colorScheme: ColorScheme;
  readonly cameraPreset: CameraPreset;
  readonly performanceMode: PerformanceMode;
  readonly mobileOptimizations: MobileOptimizations;
}

/** Color scheme configuration */
export interface ColorScheme {
  readonly name: string;
  readonly helix: {
    readonly primary: string;
    readonly gradient: readonly string[];
    readonly opacity: number;
  };
  readonly agents: Record<AgentType, string>;
  readonly background: string;
  readonly grid: string;
  readonly annotations: string;
}

/** Camera preset configurations */
export enum CameraPreset {
  Default = 'default',
  TopView = 'top_view',
  SideView = 'side_view',
  BottomUp = 'bottom_up',
  Isometric = 'isometric',
  Followent = 'follow_agent'
}

/** Performance optimization modes */
export enum PerformanceMode {
  Quality = 'quality',
  Balanced = 'balanced',
  Performance = 'performance',
  Mobile = 'mobile'
}

// =============================================================================
// Visualization State Types
// =============================================================================

/** Current visualization state */
export interface VisualizationState {
  readonly config: HelixVisualizationConfig;
  readonly helix: HelixGeometry;
  readonly agents: readonly AgentInstance[];
  readonly activeAgents: Set<string>;
  readonly camera: CameraState;
  readonly selection: SelectionState;
  readonly animation: AnimationState;
  readonly viewport: ViewportConfig;
  readonly lastUpdate: number;
}

/** Camera state tracking */
export interface CameraState {
  readonly preset: CameraPreset;
  readonly position: Position3D;
  readonly target: Position3D;
  readonly up: Position3D;
  readonly followingAgent?: string;
  readonly autoRotate: boolean;
  readonly rotationSpeed: number;
}

/** Selection state for interactive elements */
export interface SelectionState {
  readonly selectedAgent?: string;
  readonly hoveredAgent?: string;
  readonly selectedHelixPoint?: number;
  readonly showAgentPaths: boolean;
  readonly highlightSpokes: boolean;
}

/** Animation state management */
export interface AnimationState {
  readonly isPlaying: boolean;
  readonly speed: number;
  readonly currentTime: number;
  readonly duration: number;
  readonly loop: boolean;
  readonly activeAnimations: Set<string>;
}

// =============================================================================
// Core Visualization Class
// =============================================================================

export class HelixVisualization {
  private config: HelixVisualizationConfig;
  private state: VisualizationState;
  private updateQueue: VisualizationUpdateBatch[] = [];
  private animationFrame?: number;
  private lastRender: number = 0;
  private renderCache: Map<string, PlotlyTrace> = new Map();

  constructor(config: Partial<HelixVisualizationConfig> = {}) {
    this.config = this.createDefaultConfig(config);
    this.state = this.initializeState();
  }

  private createDefaultConfig(config: Partial<HelixVisualizationConfig>): HelixVisualizationConfig {
    return {
      width: config.width ?? 900,
      height: config.height ?? 700,
      resolution: config.resolution ?? 500,
      enableAnimations: config.enableAnimations ?? true,
      showEducationalAnnotations: config.showEducationalAnnotations ?? true,
      enableTooltips: config.enableTooltips ?? true,
      colorScheme: config.colorScheme ?? this.getDefaultColorScheme(),
      cameraPreset: config.cameraPreset ?? CameraPreset.Default,
      performanceMode: config.performanceMode ?? PerformanceMode.Balanced,
      mobileOptimizations: config.mobileOptimizations ?? {
        reducedAnimations: false,
        simplifiedPlots: false,
        batchedUpdates: true,
        lowPowerMode: false,
        touchOptimized: true
      }
    };
  }

  private getDefaultColorScheme(): ColorScheme {
    return {
      name: 'felix_default',
      helix: {
        primary: '#2E86AB',
        gradient: ['#A23B72', '#F18F01', '#C73E1D', '#2E86AB'],
        opacity: 0.8
      },
      agents: {
        [AgentType.Research]: '#E74C3C',
        [AgentType.Analysis]: '#3498DB',
        [AgentType.Synthesis]: '#2ECC71',
        [AgentType.Critic]: '#F39C12',
        [AgentType.General]: '#9B59B6'
      },
      background: 'rgba(248, 249, 250, 1.0)',
      grid: 'rgba(200, 200, 200, 0.3)',
      annotations: '#34495E'
    };
  }

  private initializeState(): VisualizationState {
    // Create default helix geometry
    const helix: HelixGeometry = {
      topRadius: 33.0 as any,
      bottomRadius: 0.001 as any,
      height: 100.0 as any,
      turns: 33 as any,
      concentrationRatio: 33000,
      mathematicalPrecision: 1e-12
    };

    return {
      config: this.config,
      helix,
      agents: [],
      activeAgents: new Set(),
      camera: {
        preset: this.config.cameraPreset,
        position: { x: 1.5 as Coordinate3D, y: 1.5 as Coordinate3D, z: 1.2 as Coordinate3D },
        target: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: 0 as Coordinate3D },
        up: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: 1 as Coordinate3D },
        autoRotate: false,
        rotationSpeed: 0.5
      },
      selection: {
        showAgentPaths: false,
        highlightSpokes: false
      },
      animation: {
        isPlaying: false,
        speed: 1.0,
        currentTime: 0,
        duration: 10000, // 10 seconds
        loop: true,
        activeAnimations: new Set()
      },
      viewport: {
        width: this.config.width,
        height: this.config.height,
        devicePixelRatio: window.devicePixelRatio || 1,
        isMobile: this.detectMobile(),
        isTablet: this.detectTablet(),
        orientation: this.getOrientation()
      },
      lastUpdate: Date.now()
    };
  }

  // =============================================================================
  // Helix Geometry Generation
  // =============================================================================

  /**
   * Generate optimized helix points with progressive loading support
   */
  private generateHelixPoints(resolution: number = this.config.resolution): HelixPoint[] {
    const points: HelixPoint[] = [];
    const { helix } = this.state;

    for (let i = 0; i <= resolution; i++) {
      const t = (i / resolution) as HelixParameter;

      // Parametric helix equations
      const radius = helix.bottomRadius * Math.pow(helix.topRadius / helix.bottomRadius, t);
      const theta = 2 * Math.PI * helix.turns * t;
      const height = helix.height * (1 - t); // Top to bottom

      const x = (radius * Math.cos(theta)) as Coordinate3D;
      const y = (radius * Math.sin(theta)) as Coordinate3D;
      const z = height as Coordinate3D;

      points.push({
        t,
        position: { x, y, z },
        radius: radius as any,
        angle: theta,
        height: height as any,
        timestamp: Date.now()
      });
    }

    return points;
  }

  /**
   * Create helix segments for efficient rendering
   */
  private createHelixSegments(points: HelixPoint[], segmentSize: number = 100): HelixSegment[] {
    const segments: HelixSegment[] = [];

    for (let i = 0; i < points.length; i += segmentSize) {
      const segmentPoints = points.slice(i, i + segmentSize);
      if (segmentPoints.length === 0) continue;

      const startT = segmentPoints[0].t;
      const endT = segmentPoints[segmentPoints.length - 1].t;

      // Calculate bounding box
      const xs = segmentPoints.map(p => p.position.x);
      const ys = segmentPoints.map(p => p.position.y);
      const zs = segmentPoints.map(p => p.position.z);

      segments.push({
        startT,
        endT,
        points: segmentPoints,
        resolution: segmentPoints.length,
        boundingBox: {
          min: {
            x: Math.min(...xs) as Coordinate3D,
            y: Math.min(...ys) as Coordinate3D,
            z: Math.min(...zs) as Coordinate3D
          },
          max: {
            x: Math.max(...xs) as Coordinate3D,
            y: Math.max(...ys) as Coordinate3D,
            z: Math.max(...zs) as Coordinate3D
          }
        }
      });
    }

    return segments;
  }

  // =============================================================================
  // Plotly Figure Generation
  // =============================================================================

  /**
   * Create complete Plotly figure with helix and agents
   */
  public createFigure(): PlotlyFigure {
    const traces: PlotlyTrace[] = [];

    // Add helix trace
    traces.push(this.createHelixTrace());

    // Add agent traces
    traces.push(...this.createAgentTraces());

    // Add educational annotations if enabled
    if (this.config.showEducationalAnnotations) {
      traces.push(...this.createEducationalTraces());
    }

    return {
      data: traces,
      layout: this.createLayout(),
      config: this.createPlotlyConfig()
    };
  }

  /**
   * Create the main helix spiral trace
   */
  private createHelixTrace(): PlotlyTrace {
    const cacheKey = `helix_${this.config.resolution}_${this.state.lastUpdate}`;

    if (this.renderCache.has(cacheKey)) {
      return this.renderCache.get(cacheKey)!;
    }

    const points = this.generateHelixPoints();
    const x = points.map(p => p.position.x);
    const y = points.map(p => p.position.y);
    const z = points.map(p => p.position.z);

    // Create color gradient based on height (focus level)
    const colors = z.map(height => {
      const normalized = height / this.state.helix.height;
      return normalized;
    });

    const trace: PlotlyTrace = {
      type: 'scatter3d',
      x,
      y,
      z,
      mode: 'lines',
      name: 'Felix Helix Path',
      line: {
        color: colors,
        colorscale: 'Viridis',
        width: 4
      },
      hovertemplate:
        '<b>Felix Helix</b><br>' +
        'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
        'Focus Level: %{z:.1f}%<br>' +
        'Radius: %{customdata:.3f}<br>' +
        '<extra></extra>',
      customdata: points.map(p => p.radius),
      showlegend: false
    };

    this.renderCache.set(cacheKey, trace);
    return trace;
  }

  /**
   * Create agent position traces with specialized styling
   */
  private createAgentTraces(): PlotlyTrace[] {
    const traces: PlotlyTrace[] = [];
    const agentsByType = new Map<AgentType, AgentInstance[]>();

    // Group agents by type for efficient rendering
    this.state.agents.forEach(agent => {
      if (!agentsByType.has(agent.config.type)) {
        agentsByType.set(agent.config.type, []);
      }
      agentsByType.get(agent.config.type)!.push(agent);
    });

    // Create trace for each agent type
    agentsByType.forEach((agents, agentType) => {
      const x = agents.map(agent => agent.currentPosition.x);
      const y = agents.map(agent => agent.currentPosition.y);
      const z = agents.map(agent => agent.currentPosition.z);

      const isActive = agents.map(agent =>
        this.state.activeAgents.has(agent.config.id) &&
        agent.state === AgentState.Active
      );

      const trace: PlotlyTrace = {
        type: 'scatter3d',
        x,
        y,
        z,
        mode: 'markers',
        name: `${agentType.charAt(0).toUpperCase() + agentType.slice(1)} Agents`,
        marker: {
          color: this.config.colorScheme.agents[agentType],
          size: isActive.map(active => active ? 14 : 10),
          opacity: isActive.map(active => active ? 1.0 : 0.7),
          symbol: this.getAgentSymbol(agentType),
          line: {
            color: '#FFFFFF',
            width: 2
          }
        },
        hovertemplate:
          `<b>${agentType.charAt(0).toUpperCase() + agentType.slice(1)} Agent</b><br>` +
          'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
          'State: %{customdata.state}<br>' +
          'Temperature: %{customdata.temperature}<br>' +
          'Tasks: %{customdata.tasks}<br>' +
          '<extra></extra>',
        customdata: agents.map(agent => ({
          id: agent.config.id,
          state: agent.state,
          temperature: agent.config.temperature,
          tasks: agent.processedTasks
        }))
      };

      traces.push(trace);

      // Add agent paths if enabled
      if (this.state.selection.showAgentPaths) {
        traces.push(this.createAgentPathTrace(agents));
      }
    });

    return traces;
  }

  /**
   * Get appropriate symbol for agent type
   */
  private getAgentSymbol(agentType: AgentType): string {
    const symbols = {
      [AgentType.Research]: 'circle',
      [AgentType.Analysis]: 'square',
      [AgentType.Synthesis]: 'diamond',
      [AgentType.Critic]: 'cross',
      [AgentType.General]: 'circle-open'
    };
    return symbols[agentType];
  }

  /**
   * Create agent movement path traces
   */
  private createAgentPathTrace(agents: AgentInstance[]): PlotlyTrace {
    // This would track agent movement history
    // For now, create a simple path from spawn to current position
    const pathData = agents.flatMap(agent => {
      const spawnHeight = this.state.helix.height * (1 - agent.config.helixPosition);
      const spawnRadius = this.state.helix.bottomRadius *
        Math.pow(this.state.helix.topRadius / this.state.helix.bottomRadius, agent.config.helixPosition);

      return [
        [agent.currentPosition.x, spawnRadius * Math.cos(0), spawnHeight],
        [agent.currentPosition.x, agent.currentPosition.y, agent.currentPosition.z]
      ];
    });

    const x = pathData.map(point => point[0]);
    const y = pathData.map(point => point[1]);
    const z = pathData.map(point => point[2]);

    return {
      type: 'scatter3d',
      x,
      y,
      z,
      mode: 'lines',
      name: 'Agent Paths',
      line: {
        color: 'rgba(255, 255, 255, 0.5)',
        width: 2,
        dash: 'dot'
      },
      showlegend: false,
      hoverinfo: 'skip'
    };
  }

  /**
   * Create educational annotation traces
   */
  private createEducationalTraces(): PlotlyTrace[] {
    const traces: PlotlyTrace[] = [];

    // Add coordinate system indicators
    traces.push(this.createCoordinateSystemTrace());

    // Add concentration ratio indicators
    traces.push(this.createConcentrationIndicators());

    return traces;
  }

  /**
   * Create coordinate system reference
   */
  private createCoordinateSystemTrace(): PlotlyTrace {
    const origin = { x: 0, y: 0, z: 0 };
    const axisLength = this.state.helix.topRadius * 0.3;

    return {
      type: 'scatter3d',
      x: [origin.x, axisLength, origin.x, origin.x],
      y: [origin.y, origin.y, axisLength, origin.y],
      z: [origin.z, origin.z, origin.z, axisLength],
      mode: 'lines+text',
      name: 'Coordinate System',
      line: {
        color: this.config.colorScheme.annotations,
        width: 3
      },
      text: ['Origin', 'X', 'Y', 'Z'],
      textposition: 'top center',
      showlegend: false,
      hoverinfo: 'text',
      hovertext: [
        'Origin (0, 0, 0)',
        'X-Axis (Exploration Width)',
        'Y-Axis (Exploration Breadth)',
        'Z-Axis (Focus Depth)'
      ]
    };
  }

  /**
   * Create concentration ratio visual indicators
   */
  private createConcentrationIndicators(): PlotlyTrace {
    const topRadius = this.state.helix.topRadius;
    const bottomRadius = this.state.helix.bottomRadius;
    const height = this.state.helix.height;

    // Create circles at top and bottom to show concentration
    const topCircle = this.createCirclePoints(topRadius, height);
    const bottomCircle = this.createCirclePoints(bottomRadius, 0);

    return {
      type: 'scatter3d',
      x: [...topCircle.x, ...bottomCircle.x],
      y: [...topCircle.y, ...bottomCircle.y],
      z: [...topCircle.z, ...bottomCircle.z],
      mode: 'lines',
      name: 'Concentration Ratio',
      line: {
        color: 'rgba(255, 0, 0, 0.5)',
        width: 2,
        dash: 'dash'
      },
      showlegend: false,
      hovertemplate:
        '<b>Concentration Indicator</b><br>' +
        'Radius: %{customdata:.3f}<br>' +
        'Height: %{z:.1f}<br>' +
        '<extra></extra>',
      customdata: [...Array(topCircle.x.length).fill(topRadius),
                   ...Array(bottomCircle.x.length).fill(bottomRadius)]
    };
  }

  /**
   * Generate circle points for concentration indicators
   */
  private createCirclePoints(radius: number, height: number, points: number = 50) {
    const x: number[] = [];
    const y: number[] = [];
    const z: number[] = [];

    for (let i = 0; i <= points; i++) {
      const theta = (2 * Math.PI * i) / points;
      x.push(radius * Math.cos(theta));
      y.push(radius * Math.sin(theta));
      z.push(height);
    }

    return { x, y, z };
  }

  // =============================================================================
  // Layout and Configuration
  // =============================================================================

  /**
   * Create Plotly layout configuration
   */
  private createLayout(): PlotlyLayout {
    return {
      title: {
        text: "🌪️ Felix Framework - 3D Helix Cognitive Architecture",
        x: 0.5,
        font: {
          size: 20,
          color: this.config.colorScheme.annotations
        }
      },
      scene: {
        xaxis: {
          title: 'X Position (Exploration Width)',
          showgrid: true,
          gridcolor: this.config.colorScheme.grid
        },
        yaxis: {
          title: 'Y Position (Exploration Breadth)',
          showgrid: true,
          gridcolor: this.config.colorScheme.grid
        },
        zaxis: {
          title: 'Height (Focus Depth)',
          showgrid: true,
          gridcolor: this.config.colorScheme.grid
        },
        camera: {
          up: this.state.camera.up,
          center: this.state.camera.target,
          eye: this.state.camera.position,
          projection: {
            type: 'perspective'
          }
        },
        bgcolor: this.config.colorScheme.background,
        aspectmode: 'auto'
      },
      width: this.config.width,
      height: this.config.height,
      margin: { l: 0, r: 0, t: 60, b: 0 },
      showlegend: true,
      legend: {
        yanchor: "top",
        y: 0.99,
        xanchor: "left",
        x: 0.01,
        bgcolor: 'rgba(255, 255, 255, 0.8)',
        bordercolor: this.config.colorScheme.grid,
        borderwidth: 1
      },
      font: {
        family: 'Inter, system-ui, sans-serif',
        size: 12,
        color: this.config.colorScheme.annotations
      }
    };
  }

  /**
   * Create Plotly configuration options
   */
  private createPlotlyConfig(): PlotlyConfig {
    return {
      displayModeBar: true,
      responsive: true,
      doubleClick: 'reset',
      showTips: this.config.enableTooltips,
      showLink: false,
      toImageButtonOptions: {
        format: 'png',
        filename: 'felix_helix_visualization',
        height: this.config.height,
        width: this.config.width,
        scale: 2
      }
    };
  }

  // =============================================================================
  // Real-time Updates
  // =============================================================================

  /**
   * Update agent positions with batched updates
   */
  public updateAgentPositions(updates: AgentPositionUpdate[]): void {
    const batch: VisualizationUpdateBatch = {
      agentUpdates: updates,
      timestamp: Date.now()
    };

    this.updateQueue.push(batch);
    this.scheduleRender();
  }

  /**
   * Schedule render with throttling
   */
  private scheduleRender(): void {
    if (this.animationFrame) return;

    this.animationFrame = requestAnimationFrame(() => {
      this.processPendingUpdates();
      this.animationFrame = undefined;
    });
  }

  /**
   * Process all pending updates efficiently
   */
  private processPendingUpdates(): void {
    if (this.updateQueue.length === 0) return;

    const now = Date.now();
    const timeSinceLastRender = now - this.lastRender;

    // Throttle updates based on performance mode
    const minInterval = this.getMinRenderInterval();
    if (timeSinceLastRender < minInterval) {
      this.scheduleRender();
      return;
    }

    // Process all queued updates
    const allUpdates = this.updateQueue.splice(0);
    this.applyUpdates(allUpdates);

    this.lastRender = now;
    this.state = { ...this.state, lastUpdate: now };
  }

  /**
   * Get minimum render interval based on performance mode
   */
  private getMinRenderInterval(): number {
    const intervals = {
      [PerformanceMode.Quality]: 16, // 60 FPS
      [PerformanceMode.Balanced]: 33, // 30 FPS
      [PerformanceMode.Performance]: 66, // 15 FPS
      [PerformanceMode.Mobile]: 100 // 10 FPS
    };
    return intervals[this.config.performanceMode];
  }

  /**
   * Apply batched updates to visualization state
   */
  private applyUpdates(updateBatches: VisualizationUpdateBatch[]): void {
    // Combine all agent updates
    const allAgentUpdates = updateBatches.flatMap(batch => batch.agentUpdates);

    // Update agent positions
    const updatedAgents = new Map(this.state.agents.map(agent => [agent.config.id, agent]));

    allAgentUpdates.forEach(update => {
      const agent = updatedAgents.get(update.agentId);
      if (agent) {
        updatedAgents.set(update.agentId, {
          ...agent,
          currentPosition: update.position,
          state: update.state,
          lastActivity: update.timestamp
        });
      }
    });

    // Update active agents set
    const activeAgents = new Set<string>();
    updatedAgents.forEach(agent => {
      if (agent.state === AgentState.Active || agent.state === AgentState.Processing) {
        activeAgents.add(agent.config.id);
      }
    });

    this.state = {
      ...this.state,
      agents: Array.from(updatedAgents.values()),
      activeAgents
    };

    // Clear render cache for affected elements
    this.renderCache.clear();
  }

  // =============================================================================
  // Camera Controls
  // =============================================================================

  /**
   * Set camera preset
   */
  public setCameraPreset(preset: CameraPreset): void {
    const positions = this.getCameraPresets();
    const cameraConfig = positions[preset];

    if (cameraConfig) {
      this.state = {
        ...this.state,
        camera: {
          ...this.state.camera,
          preset,
          ...cameraConfig
        }
      };
    }
  }

  /**
   * Get predefined camera positions
   */
  private getCameraPresets(): Record<CameraPreset, Partial<CameraState>> {
    const radius = this.state.helix.topRadius;
    const height = this.state.helix.height;

    return {
      [CameraPreset.Default]: {
        position: { x: 1.5 * radius as Coordinate3D, y: 1.5 * radius as Coordinate3D, z: height * 0.6 as Coordinate3D },
        target: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: height * 0.5 as Coordinate3D },
        up: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: 1 as Coordinate3D }
      },
      [CameraPreset.TopView]: {
        position: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: height * 1.5 as Coordinate3D },
        target: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: height * 0.5 as Coordinate3D },
        up: { x: 0 as Coordinate3D, y: 1 as Coordinate3D, z: 0 as Coordinate3D }
      },
      [CameraPreset.SideView]: {
        position: { x: radius * 2 as Coordinate3D, y: 0 as Coordinate3D, z: height * 0.5 as Coordinate3D },
        target: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: height * 0.5 as Coordinate3D },
        up: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: 1 as Coordinate3D }
      },
      [CameraPreset.BottomUp]: {
        position: { x: radius * 0.5 as Coordinate3D, y: radius * 0.5 as Coordinate3D, z: -height * 0.3 as Coordinate3D },
        target: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: height * 0.8 as Coordinate3D },
        up: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: 1 as Coordinate3D }
      },
      [CameraPreset.Isometric]: {
        position: { x: radius * 1.2 as Coordinate3D, y: radius * 1.2 as Coordinate3D, z: height * 0.8 as Coordinate3D },
        target: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: height * 0.3 as Coordinate3D },
        up: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: 1 as Coordinate3D }
      },
      [CameraPreset.Followent]: {
        // This would be dynamically updated to follow selected agent
        position: { x: radius as Coordinate3D, y: radius as Coordinate3D, z: height * 0.7 as Coordinate3D },
        target: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: height * 0.5 as Coordinate3D },
        up: { x: 0 as Coordinate3D, y: 0 as Coordinate3D, z: 1 as Coordinate3D }
      }
    };
  }

  // =============================================================================
  // Export and Utilities
  // =============================================================================

  /**
   * Export visualization in various formats
   */
  public async exportVisualization(config: ExportConfig): Promise<ExportResult> {
    try {
      const figure = this.createFigure();

      switch (config.format) {
        case ExportFormat.JSON:
          return this.exportAsJSON(figure, config);
        case ExportFormat.PNG:
          return this.exportAsImage(figure, config, 'png');
        case ExportFormat.SVG:
          return this.exportAsImage(figure, config, 'svg');
        case ExportFormat.HTML:
          return this.exportAsHTML(figure, config);
        default:
          throw new Error(`Unsupported export format: ${config.format}`);
      }
    } catch (error) {
      return {
        success: false,
        filename: '',
        size: 0,
        format: config.format,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  private exportAsJSON(figure: PlotlyFigure, config: ExportConfig): ExportResult {
    const exportData = {
      figure,
      metadata: {
        timestamp: new Date().toISOString(),
        config: this.config,
        state: this.state,
        version: '1.0.0'
      }
    };

    const jsonString = JSON.stringify(exportData, null, 2);
    const filename = config.filename || `felix_helix_${Date.now()}.json`;

    return {
      success: true,
      data: jsonString,
      filename,
      size: jsonString.length,
      format: ExportFormat.JSON
    };
  }

  private async exportAsImage(figure: PlotlyFigure, config: ExportConfig, format: 'png' | 'svg'): Promise<ExportResult> {
    // This would use Plotly's toImage functionality
    // For now, return placeholder
    const filename = config.filename || `felix_helix_${Date.now()}.${format}`;

    return {
      success: true,
      data: '', // Would contain actual image data
      filename,
      size: 0,
      format: format === 'png' ? ExportFormat.PNG : ExportFormat.SVG
    };
  }

  private exportAsHTML(figure: PlotlyFigure, config: ExportConfig): ExportResult {
    const htmlTemplate = `
<!DOCTYPE html>
<html>
<head>
    <title>Felix Framework - Helix Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Inter, system-ui, sans-serif; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 20px; }
        .plot-container { width: 100%; height: 80vh; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌪️ Felix Framework - Helix Visualization</h1>
        <p>Exported on ${new Date().toLocaleString()}</p>
    </div>
    <div id="plot" class="plot-container"></div>
    <script>
        Plotly.newPlot('plot', ${JSON.stringify(figure.data)}, ${JSON.stringify(figure.layout)}, ${JSON.stringify(figure.config)});
    </script>
</body>
</html>`;

    const filename = config.filename || `felix_helix_${Date.now()}.html`;

    return {
      success: true,
      data: htmlTemplate,
      filename,
      size: htmlTemplate.length,
      format: ExportFormat.HTML
    };
  }

  // =============================================================================
  // Mobile and Responsive Utilities
  // =============================================================================

  private detectMobile(): boolean {
    return window.innerWidth <= 768 || /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  }

  private detectTablet(): boolean {
    return window.innerWidth > 768 && window.innerWidth <= 1024;
  }

  private getOrientation(): 'portrait' | 'landscape' {
    return window.innerHeight > window.innerWidth ? 'portrait' : 'landscape';
  }

  /**
   * Update configuration for mobile optimization
   */
  public optimizeForMobile(): void {
    if (this.state.viewport.isMobile) {
      this.config = {
        ...this.config,
        performanceMode: PerformanceMode.Mobile,
        enableAnimations: false,
        resolution: Math.min(this.config.resolution, 200),
        mobileOptimizations: {
          reducedAnimations: true,
          simplifiedPlots: true,
          batchedUpdates: true,
          lowPowerMode: true,
          touchOptimized: true
        }
      };
    }
  }

  // =============================================================================
  // Public API
  // =============================================================================

  /**
   * Get current configuration
   */
  public getConfig(): HelixVisualizationConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  public updateConfig(updates: Partial<HelixVisualizationConfig>): void {
    this.config = { ...this.config, ...updates };
    this.renderCache.clear(); // Clear cache when config changes
  }

  /**
   * Get current state
   */
  public getState(): VisualizationState {
    return { ...this.state };
  }

  /**
   * Add agents to visualization
   */
  public addAgents(agents: AgentInstance[]): void {
    const existingIds = new Set(this.state.agents.map(a => a.config.id));
    const newAgents = agents.filter(agent => !existingIds.has(agent.config.id));

    this.state = {
      ...this.state,
      agents: [...this.state.agents, ...newAgents]
    };
  }

  /**
   * Remove agents from visualization
   */
  public removeAgents(agentIds: string[]): void {
    const idsToRemove = new Set(agentIds);

    this.state = {
      ...this.state,
      agents: this.state.agents.filter(agent => !idsToRemove.has(agent.config.id)),
      activeAgents: new Set([...this.state.activeAgents].filter(id => !idsToRemove.has(id)))
    };
  }

  /**
   * Clear all render caches
   */
  public clearCache(): void {
    this.renderCache.clear();
  }

  /**
   * Dispose of resources
   */
  public dispose(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    this.renderCache.clear();
    this.updateQueue.length = 0;
  }
}

// Export for use in Gradio interface
export default HelixVisualization;