/**
 * Felix Framework - Interactive Parameter Controls
 *
 * Type-safe parameter controls for Felix Framework configuration with
 * real-time validation, constraint enforcement, and mobile-optimized
 * input handling for helix geometry, agent configuration, and system settings.
 *
 * Features:
 * - Type-safe parameter validation
 * - Real-time constraint checking
 * - Dependency resolution between parameters
 * - Mobile-optimized input controls
 * - Undo/redo functionality
 * - Parameter presets and templates
 * - Export/import configurations
 * - Performance impact prediction
 *
 * @version 1.0.0
 * @author Felix Framework Team
 */

import {
  GradioSlider,
  GradioTextbox,
  GradioCheckboxGroup,
  GradioDropdown,
  GradioCheckbox,
  ValidationResult,
  ValidationError,
  ValidationWarning
} from '../types/gradio-interface';

import {
  HelixGeometry,
  AgentConfig,
  AgentType,
  TaskRequirements,
  PerformanceMetrics,
  Radius,
  Turns,
  Height,
  HelixParameter
} from '../types/felix-core';

// =============================================================================
// Parameter Control Types
// =============================================================================

/** Parameter control configuration */
export interface ParameterControlConfig {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
  readonly type: ParameterType;
  readonly constraints: ParameterConstraints;
  readonly defaultValue: unknown;
  readonly dependencies?: readonly string[];
  readonly mobileOptimized?: boolean;
  readonly performanceImpact?: PerformanceImpact;
}

/** Parameter types */
export enum ParameterType {
  Number = 'number',
  Integer = 'integer',
  Float = 'float',
  Boolean = 'boolean',
  String = 'string',
  Enum = 'enum',
  Range = 'range',
  MultiSelect = 'multi_select'
}

/** Parameter constraints */
export interface ParameterConstraints {
  readonly min?: number;
  readonly max?: number;
  readonly step?: number;
  readonly precision?: number;
  readonly options?: readonly string[];
  readonly pattern?: RegExp;
  readonly required?: boolean;
  readonly customValidator?: (value: unknown) => ValidationResult;
}

/** Performance impact levels */
export enum PerformanceImpact {
  None = 'none',
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Critical = 'critical'
}

// =============================================================================
// Helix Geometry Parameter Controls
// =============================================================================

/** Helix geometry parameter configuration */
export interface HelixGeometryParams {
  readonly topRadius: number;
  readonly bottomRadius: number;
  readonly height: number;
  readonly turns: number;
  readonly resolution: number;
  readonly mathematicalPrecision: number;
}

/** Helix parameter control definitions */
export const HELIX_PARAMETER_CONTROLS: Record<keyof HelixGeometryParams, ParameterControlConfig> = {
  topRadius: {
    id: 'helix_top_radius',
    label: 'Top Radius',
    description: 'Radius at the top of the helix (broad exploration phase)',
    type: ParameterType.Float,
    constraints: {
      min: 0.1,
      max: 100.0,
      step: 0.1,
      precision: 3,
      required: true,
      customValidator: (value) => validateRadius(value as number, 'top')
    },
    defaultValue: 33.0,
    dependencies: ['bottomRadius'],
    performanceImpact: PerformanceImpact.Medium
  },

  bottomRadius: {
    id: 'helix_bottom_radius',
    label: 'Bottom Radius',
    description: 'Radius at the bottom of the helix (focused synthesis phase)',
    type: ParameterType.Float,
    constraints: {
      min: 0.001,
      max: 10.0,
      step: 0.001,
      precision: 6,
      required: true,
      customValidator: (value) => validateRadius(value as number, 'bottom')
    },
    defaultValue: 0.001,
    dependencies: ['topRadius'],
    performanceImpact: PerformanceImpact.Low
  },

  height: {
    id: 'helix_height',
    label: 'Helix Height',
    description: 'Total height of the helix structure',
    type: ParameterType.Float,
    constraints: {
      min: 10.0,
      max: 1000.0,
      step: 1.0,
      precision: 1,
      required: true
    },
    defaultValue: 100.0,
    performanceImpact: PerformanceImpact.Low
  },

  turns: {
    id: 'helix_turns',
    label: 'Number of Turns',
    description: 'Complete rotations in the helix (affects agent path complexity)',
    type: ParameterType.Integer,
    constraints: {
      min: 1,
      max: 100,
      step: 1,
      required: true,
      customValidator: validateTurns
    },
    defaultValue: 33,
    dependencies: ['resolution'],
    performanceImpact: PerformanceImpact.High
  },

  resolution: {
    id: 'helix_resolution',
    label: 'Visualization Resolution',
    description: 'Number of points used for helix visualization (affects rendering performance)',
    type: ParameterType.Integer,
    constraints: {
      min: 50,
      max: 2000,
      step: 50,
      required: true
    },
    defaultValue: 500,
    mobileOptimized: true,
    performanceImpact: PerformanceImpact.Critical
  },

  mathematicalPrecision: {
    id: 'helix_precision',
    label: 'Mathematical Precision',
    description: 'Precision threshold for mathematical calculations (scientific notation)',
    type: ParameterType.Float,
    constraints: {
      min: 1e-15,
      max: 1e-6,
      step: 1e-15,
      precision: 15
    },
    defaultValue: 1e-12,
    performanceImpact: PerformanceImpact.None
  }
};

// =============================================================================
// Agent Configuration Parameter Controls
// =============================================================================

/** Agent configuration parameters */
export interface AgentConfigParams {
  readonly maxAgents: number;
  readonly spawnInterval: number;
  readonly agentTypes: readonly AgentType[];
  readonly temperatureSettings: Record<AgentType, number>;
  readonly lifetimeSettings: Record<AgentType, number>;
  readonly enableAdaptiveSpawning: boolean;
  readonly enableLoadBalancing: boolean;
}

/** Agent parameter control definitions */
export const AGENT_PARAMETER_CONTROLS: Record<string, ParameterControlConfig> = {
  maxAgents: {
    id: 'agent_max_count',
    label: 'Maximum Agents',
    description: 'Maximum number of concurrent agents (affects memory usage)',
    type: ParameterType.Integer,
    constraints: {
      min: 1,
      max: 100,
      step: 1,
      required: true,
      customValidator: validateMaxAgents
    },
    defaultValue: 20,
    performanceImpact: PerformanceImpact.Critical
  },

  spawnInterval: {
    id: 'agent_spawn_interval',
    label: 'Spawn Interval (ms)',
    description: 'Time between agent spawns in milliseconds',
    type: ParameterType.Integer,
    constraints: {
      min: 100,
      max: 10000,
      step: 100,
      required: true
    },
    defaultValue: 1000,
    performanceImpact: PerformanceImpact.Medium
  },

  agentTypes: {
    id: 'agent_types_enabled',
    label: 'Enabled Agent Types',
    description: 'Select which agent types are available for spawning',
    type: ParameterType.MultiSelect,
    constraints: {
      options: Object.values(AgentType),
      required: true,
      customValidator: validateAgentTypes
    },
    defaultValue: [AgentType.Research, AgentType.Analysis, AgentType.Synthesis],
    performanceImpact: PerformanceImpact.Medium
  },

  enableAdaptiveSpawning: {
    id: 'agent_adaptive_spawning',
    label: 'Adaptive Spawning',
    description: 'Automatically adjust agent spawning based on task complexity',
    type: ParameterType.Boolean,
    constraints: {
      required: false
    },
    defaultValue: true,
    performanceImpact: PerformanceImpact.Low
  },

  enableLoadBalancing: {
    id: 'agent_load_balancing',
    label: 'Load Balancing',
    description: 'Distribute tasks evenly across available agents',
    type: ParameterType.Boolean,
    constraints: {
      required: false
    },
    defaultValue: true,
    performanceImpact: PerformanceImpact.Low
  }
};

// =============================================================================
// System Performance Parameter Controls
// =============================================================================

/** System performance parameters */
export interface SystemPerformanceParams {
  readonly updateInterval: number;
  readonly renderingMode: RenderingMode;
  readonly memoryLimit: number;
  readonly enableGPU: boolean;
  readonly compressionLevel: number;
  readonly cacheSize: number;
  readonly logLevel: LogLevel;
}

/** Rendering modes */
export enum RenderingMode {
  Quality = 'quality',
  Balanced = 'balanced',
  Performance = 'performance',
  Mobile = 'mobile'
}

/** Log levels */
export enum LogLevel {
  Debug = 'debug',
  Info = 'info',
  Warning = 'warning',
  Error = 'error'
}

/** System parameter control definitions */
export const SYSTEM_PARAMETER_CONTROLS: Record<string, ParameterControlConfig> = {
  updateInterval: {
    id: 'system_update_interval',
    label: 'Update Interval (ms)',
    description: 'How often the system updates visualization and metrics',
    type: ParameterType.Integer,
    constraints: {
      min: 16,
      max: 5000,
      step: 16,
      required: true
    },
    defaultValue: 500,
    mobileOptimized: true,
    performanceImpact: PerformanceImpact.High
  },

  renderingMode: {
    id: 'system_rendering_mode',
    label: 'Rendering Mode',
    description: 'Balance between visual quality and performance',
    type: ParameterType.Enum,
    constraints: {
      options: Object.values(RenderingMode),
      required: true
    },
    defaultValue: RenderingMode.Balanced,
    mobileOptimized: true,
    performanceImpact: PerformanceImpact.Critical
  },

  memoryLimit: {
    id: 'system_memory_limit',
    label: 'Memory Limit (MB)',
    description: 'Maximum memory usage for the application',
    type: ParameterType.Integer,
    constraints: {
      min: 100,
      max: 8000,
      step: 100,
      required: true
    },
    defaultValue: 1000,
    performanceImpact: PerformanceImpact.High
  },

  enableGPU: {
    id: 'system_enable_gpu',
    label: 'GPU Acceleration',
    description: 'Use GPU for compute-intensive operations when available',
    type: ParameterType.Boolean,
    constraints: {
      required: false,
      customValidator: validateGPUAvailability
    },
    defaultValue: true,
    performanceImpact: PerformanceImpact.Critical
  },

  compressionLevel: {
    id: 'system_compression_level',
    label: 'Data Compression',
    description: 'Level of compression for data transfer (0=none, 9=maximum)',
    type: ParameterType.Integer,
    constraints: {
      min: 0,
      max: 9,
      step: 1,
      required: true
    },
    defaultValue: 3,
    performanceImpact: PerformanceImpact.Medium
  },

  cacheSize: {
    id: 'system_cache_size',
    label: 'Cache Size (MB)',
    description: 'Size of rendering and data cache',
    type: ParameterType.Integer,
    constraints: {
      min: 10,
      max: 500,
      step: 10,
      required: true
    },
    defaultValue: 100,
    performanceImpact: PerformanceImpact.Medium
  },

  logLevel: {
    id: 'system_log_level',
    label: 'Log Level',
    description: 'Minimum level for system logging',
    type: ParameterType.Enum,
    constraints: {
      options: Object.values(LogLevel),
      required: true
    },
    defaultValue: LogLevel.Info,
    performanceImpact: PerformanceImpact.Low
  }
};

// =============================================================================
// Parameter Control Manager
// =============================================================================

export class ParameterControlManager {
  private currentValues: Map<string, unknown> = new Map();
  private parameterHistory: ParameterHistoryEntry[] = [];
  private validationResults: Map<string, ValidationResult> = new Map();
  private dependencyGraph: Map<string, Set<string>> = new Map();
  private changeListeners: Map<string, Set<ParameterChangeListener>> = new Map();

  constructor(private readonly isMobile: boolean = false) {
    this.initializeParameters();
    this.buildDependencyGraph();
  }

  // =============================================================================
  // Parameter Management
  // =============================================================================

  /**
   * Initialize parameters with default values
   */
  private initializeParameters(): void {
    const allControls = {
      ...HELIX_PARAMETER_CONTROLS,
      ...AGENT_PARAMETER_CONTROLS,
      ...SYSTEM_PARAMETER_CONTROLS
    };

    Object.values(allControls).forEach(control => {
      let defaultValue = control.defaultValue;

      // Apply mobile optimizations
      if (this.isMobile && control.mobileOptimized) {
        defaultValue = this.getMobileOptimizedValue(control);
      }

      this.currentValues.set(control.id, defaultValue);
    });
  }

  /**
   * Get mobile-optimized parameter values
   */
  private getMobileOptimizedValue(control: ParameterControlConfig): unknown {
    switch (control.id) {
      case 'helix_resolution':
        return Math.min(control.defaultValue as number, 200);
      case 'system_update_interval':
        return Math.max(control.defaultValue as number, 1000);
      case 'system_rendering_mode':
        return RenderingMode.Mobile;
      default:
        return control.defaultValue;
    }
  }

  /**
   * Build dependency graph for parameter relationships
   */
  private buildDependencyGraph(): void {
    const allControls = {
      ...HELIX_PARAMETER_CONTROLS,
      ...AGENT_PARAMETER_CONTROLS,
      ...SYSTEM_PARAMETER_CONTROLS
    };

    Object.values(allControls).forEach(control => {
      if (control.dependencies) {
        control.dependencies.forEach(depId => {
          if (!this.dependencyGraph.has(depId)) {
            this.dependencyGraph.set(depId, new Set());
          }
          this.dependencyGraph.get(depId)!.add(control.id);
        });
      }
    });
  }

  /**
   * Set parameter value with validation and dependency checking
   */
  public setParameter(parameterId: string, value: unknown): ValidationResult {
    const control = this.getParameterControl(parameterId);
    if (!control) {
      return {
        valid: false,
        errors: [{ field: parameterId, message: 'Parameter not found', code: 'NOT_FOUND' }],
        warnings: []
      };
    }

    // Validate the value
    const validationResult = this.validateParameter(control, value);
    if (!validationResult.valid) {
      this.validationResults.set(parameterId, validationResult);
      return validationResult;
    }

    // Store previous value for history
    const previousValue = this.currentValues.get(parameterId);

    // Update value
    this.currentValues.set(parameterId, value);
    this.validationResults.set(parameterId, validationResult);

    // Add to history
    this.addToHistory(parameterId, previousValue, value);

    // Validate dependent parameters
    this.validateDependencies(parameterId);

    // Notify listeners
    this.notifyChangeListeners(parameterId, value, previousValue);

    return validationResult;
  }

  /**
   * Get parameter value
   */
  public getParameter<T>(parameterId: string): T | undefined {
    return this.currentValues.get(parameterId) as T;
  }

  /**
   * Get all parameter values
   */
  public getAllParameters(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    this.currentValues.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  }

  /**
   * Reset parameters to defaults
   */
  public resetToDefaults(): void {
    this.currentValues.clear();
    this.validationResults.clear();
    this.parameterHistory.length = 0;
    this.initializeParameters();
  }

  // =============================================================================
  // Validation
  // =============================================================================

  /**
   * Validate a parameter value
   */
  private validateParameter(control: ParameterControlConfig, value: unknown): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Type validation
    if (!this.validateType(control.type, value)) {
      errors.push({
        field: control.id,
        message: `Invalid type. Expected ${control.type}`,
        code: 'INVALID_TYPE'
      });
      return { valid: false, errors, warnings };
    }

    // Constraint validation
    const constraintResult = this.validateConstraints(control.constraints, value);
    errors.push(...constraintResult.errors);
    warnings.push(...constraintResult.warnings);

    // Custom validation
    if (control.constraints.customValidator) {
      const customResult = control.constraints.customValidator(value);
      errors.push(...customResult.errors);
      warnings.push(...customResult.warnings);
    }

    // Performance impact warning
    if (control.performanceImpact === PerformanceImpact.Critical) {
      warnings.push({
        field: control.id,
        message: 'This parameter has critical performance impact',
        suggestion: 'Consider the effect on system performance'
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate parameter type
   */
  private validateType(expectedType: ParameterType, value: unknown): boolean {
    switch (expectedType) {
      case ParameterType.Number:
      case ParameterType.Float:
      case ParameterType.Integer:
        return typeof value === 'number' && !isNaN(value);
      case ParameterType.Boolean:
        return typeof value === 'boolean';
      case ParameterType.String:
        return typeof value === 'string';
      case ParameterType.Enum:
        return typeof value === 'string';
      case ParameterType.MultiSelect:
        return Array.isArray(value);
      case ParameterType.Range:
        return Array.isArray(value) && value.length === 2 &&
               typeof value[0] === 'number' && typeof value[1] === 'number';
      default:
        return false;
    }
  }

  /**
   * Validate parameter constraints
   */
  private validateConstraints(constraints: ParameterConstraints, value: unknown): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Required validation
    if (constraints.required && (value === null || value === undefined || value === '')) {
      errors.push({
        field: 'value',
        message: 'This parameter is required',
        code: 'REQUIRED'
      });
    }

    // Numeric constraints
    if (typeof value === 'number') {
      if (constraints.min !== undefined && value < constraints.min) {
        errors.push({
          field: 'value',
          message: `Value must be at least ${constraints.min}`,
          code: 'MIN_VALUE'
        });
      }

      if (constraints.max !== undefined && value > constraints.max) {
        errors.push({
          field: 'value',
          message: `Value must be at most ${constraints.max}`,
          code: 'MAX_VALUE'
        });
      }

      if (constraints.step !== undefined) {
        const remainder = (value - (constraints.min || 0)) % constraints.step;
        if (Math.abs(remainder) > 1e-10) {
          errors.push({
            field: 'value',
            message: `Value must be a multiple of ${constraints.step}`,
            code: 'STEP_VALUE'
          });
        }
      }
    }

    // Options validation
    if (constraints.options && typeof value === 'string') {
      if (!constraints.options.includes(value)) {
        errors.push({
          field: 'value',
          message: `Value must be one of: ${constraints.options.join(', ')}`,
          code: 'INVALID_OPTION'
        });
      }
    }

    // Pattern validation
    if (constraints.pattern && typeof value === 'string') {
      if (!constraints.pattern.test(value)) {
        errors.push({
          field: 'value',
          message: 'Value does not match required pattern',
          code: 'PATTERN_MISMATCH'
        });
      }
    }

    return { valid: errors.length === 0, errors, warnings };
  }

  /**
   * Validate dependent parameters when a parameter changes
   */
  private validateDependencies(changedParameterId: string): void {
    const dependents = this.dependencyGraph.get(changedParameterId);
    if (!dependents) return;

    dependents.forEach(dependentId => {
      const control = this.getParameterControl(dependentId);
      const value = this.currentValues.get(dependentId);

      if (control && value !== undefined) {
        const result = this.validateParameter(control, value);
        this.validationResults.set(dependentId, result);
      }
    });
  }

  // =============================================================================
  // Gradio Integration
  // =============================================================================

  /**
   * Create Gradio slider component
   */
  public createGradioSlider(parameterId: string): GradioSlider {
    const control = this.getParameterControl(parameterId);
    if (!control) throw new Error(`Parameter ${parameterId} not found`);

    const value = this.getParameter<number>(parameterId) || 0;
    const validation = this.validationResults.get(parameterId);

    return {
      value,
      label: control.label,
      info: control.description,
      visible: true,
      interactive: true,
      minimum: control.constraints.min || 0,
      maximum: control.constraints.max || 100,
      step: control.constraints.step || 1,
      // Additional Gradio-specific properties would be added here
    };
  }

  /**
   * Create Gradio textbox component
   */
  public createGradioTextbox(parameterId: string): GradioTextbox {
    const control = this.getParameterControl(parameterId);
    if (!control) throw new Error(`Parameter ${parameterId} not found`);

    const value = this.getParameter<string>(parameterId) || '';

    return {
      value,
      label: control.label,
      info: control.description,
      visible: true,
      interactive: true,
      placeholder: `Enter ${control.label.toLowerCase()}...`,
      type: control.type === ParameterType.String ? 'text' : 'text'
    };
  }

  /**
   * Create Gradio checkbox group component
   */
  public createGradioCheckboxGroup(parameterId: string): GradioCheckboxGroup {
    const control = this.getParameterControl(parameterId);
    if (!control) throw new Error(`Parameter ${parameterId} not found`);

    const value = this.getParameter<readonly string[]>(parameterId) || [];

    return {
      value,
      label: control.label,
      info: control.description,
      visible: true,
      interactive: true,
      choices: control.constraints.options || []
    };
  }

  /**
   * Create Gradio dropdown component
   */
  public createGradioDropdown(parameterId: string): GradioDropdown {
    const control = this.getParameterControl(parameterId);
    if (!control) throw new Error(`Parameter ${parameterId} not found`);

    const value = this.getParameter<string>(parameterId) || '';

    return {
      value,
      label: control.label,
      info: control.description,
      visible: true,
      interactive: true,
      choices: control.constraints.options || []
    };
  }

  // =============================================================================
  // Presets and Templates
  // =============================================================================

  /**
   * Create parameter preset
   */
  public createPreset(name: string, description: string): ParameterPreset {
    return {
      id: `preset_${Date.now()}`,
      name,
      description,
      parameters: this.getAllParameters(),
      createdAt: Date.now(),
      version: '1.0.0'
    };
  }

  /**
   * Apply parameter preset
   */
  public applyPreset(preset: ParameterPreset): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    Object.entries(preset.parameters).forEach(([parameterId, value]) => {
      const result = this.setParameter(parameterId, value);
      errors.push(...result.errors);
      warnings.push(...result.warnings);
    });

    return { valid: errors.length === 0, errors, warnings };
  }

  /**
   * Get built-in presets
   */
  public getBuiltInPresets(): ParameterPreset[] {
    return [
      {
        id: 'performance_optimized',
        name: 'Performance Optimized',
        description: 'Settings optimized for maximum performance',
        parameters: {
          'helix_resolution': 200,
          'system_rendering_mode': RenderingMode.Performance,
          'system_update_interval': 1000,
          'agent_max_count': 10
        },
        createdAt: Date.now(),
        version: '1.0.0'
      },
      {
        id: 'quality_focused',
        name: 'Quality Focused',
        description: 'Settings optimized for visual quality',
        parameters: {
          'helix_resolution': 1000,
          'system_rendering_mode': RenderingMode.Quality,
          'system_update_interval': 250,
          'agent_max_count': 20
        },
        createdAt: Date.now(),
        version: '1.0.0'
      },
      {
        id: 'mobile_optimized',
        name: 'Mobile Optimized',
        description: 'Settings optimized for mobile devices',
        parameters: {
          'helix_resolution': 150,
          'system_rendering_mode': RenderingMode.Mobile,
          'system_update_interval': 2000,
          'agent_max_count': 5,
          'system_compression_level': 6
        },
        createdAt: Date.now(),
        version: '1.0.0'
      }
    ];
  }

  // =============================================================================
  // History and Undo/Redo
  // =============================================================================

  /**
   * Add parameter change to history
   */
  private addToHistory(parameterId: string, oldValue: unknown, newValue: unknown): void {
    this.parameterHistory.push({
      parameterId,
      oldValue,
      newValue,
      timestamp: Date.now()
    });

    // Limit history size
    if (this.parameterHistory.length > 100) {
      this.parameterHistory.shift();
    }
  }

  /**
   * Undo last parameter change
   */
  public undo(): boolean {
    if (this.parameterHistory.length === 0) return false;

    const lastChange = this.parameterHistory.pop()!;
    this.currentValues.set(lastChange.parameterId, lastChange.oldValue);

    // Re-validate
    const control = this.getParameterControl(lastChange.parameterId);
    if (control) {
      const result = this.validateParameter(control, lastChange.oldValue);
      this.validationResults.set(lastChange.parameterId, result);
    }

    return true;
  }

  /**
   * Get parameter change history
   */
  public getHistory(): readonly ParameterHistoryEntry[] {
    return [...this.parameterHistory];
  }

  // =============================================================================
  // Event Handling
  // =============================================================================

  /**
   * Add parameter change listener
   */
  public addChangeListener(parameterId: string, listener: ParameterChangeListener): void {
    if (!this.changeListeners.has(parameterId)) {
      this.changeListeners.set(parameterId, new Set());
    }
    this.changeListeners.get(parameterId)!.add(listener);
  }

  /**
   * Remove parameter change listener
   */
  public removeChangeListener(parameterId: string, listener: ParameterChangeListener): boolean {
    const listeners = this.changeListeners.get(parameterId);
    return listeners ? listeners.delete(listener) : false;
  }

  /**
   * Notify change listeners
   */
  private notifyChangeListeners(parameterId: string, newValue: unknown, oldValue: unknown): void {
    const listeners = this.changeListeners.get(parameterId);
    if (!listeners) return;

    listeners.forEach(listener => {
      try {
        listener(parameterId, newValue, oldValue);
      } catch (error) {
        console.error('Error in parameter change listener:', error);
      }
    });
  }

  // =============================================================================
  // Utility Methods
  // =============================================================================

  /**
   * Get parameter control configuration
   */
  private getParameterControl(parameterId: string): ParameterControlConfig | undefined {
    const allControls = {
      ...HELIX_PARAMETER_CONTROLS,
      ...AGENT_PARAMETER_CONTROLS,
      ...SYSTEM_PARAMETER_CONTROLS
    };

    return Object.values(allControls).find(control => control.id === parameterId);
  }

  /**
   * Get validation result for parameter
   */
  public getValidationResult(parameterId: string): ValidationResult | undefined {
    return this.validationResults.get(parameterId);
  }

  /**
   * Check if all parameters are valid
   */
  public isAllValid(): boolean {
    return Array.from(this.validationResults.values()).every(result => result.valid);
  }

  /**
   * Get performance impact summary
   */
  public getPerformanceImpact(): PerformanceImpactSummary {
    const impacts = Array.from(this.currentValues.keys())
      .map(id => this.getParameterControl(id))
      .filter(control => control?.performanceImpact)
      .map(control => control!.performanceImpact!);

    return {
      overall: this.calculateOverallImpact(impacts),
      breakdown: this.groupImpactsByLevel(impacts),
      recommendations: this.getPerformanceRecommendations()
    };
  }

  private calculateOverallImpact(impacts: PerformanceImpact[]): PerformanceImpact {
    if (impacts.includes(PerformanceImpact.Critical)) return PerformanceImpact.Critical;
    if (impacts.includes(PerformanceImpact.High)) return PerformanceImpact.High;
    if (impacts.includes(PerformanceImpact.Medium)) return PerformanceImpact.Medium;
    if (impacts.includes(PerformanceImpact.Low)) return PerformanceImpact.Low;
    return PerformanceImpact.None;
  }

  private groupImpactsByLevel(impacts: PerformanceImpact[]): Record<PerformanceImpact, number> {
    const breakdown = {
      [PerformanceImpact.None]: 0,
      [PerformanceImpact.Low]: 0,
      [PerformanceImpact.Medium]: 0,
      [PerformanceImpact.High]: 0,
      [PerformanceImpact.Critical]: 0
    };

    impacts.forEach(impact => {
      breakdown[impact]++;
    });

    return breakdown;
  }

  private getPerformanceRecommendations(): string[] {
    const recommendations: string[] = [];

    if (this.isMobile) {
      recommendations.push('Consider using the Mobile Optimized preset for better performance');
    }

    const resolution = this.getParameter<number>('helix_resolution');
    if (resolution && resolution > 1000) {
      recommendations.push('High resolution may impact rendering performance');
    }

    const maxAgents = this.getParameter<number>('agent_max_count');
    if (maxAgents && maxAgents > 20) {
      recommendations.push('High agent count may impact system performance');
    }

    return recommendations;
  }
}

// =============================================================================
// Supporting Types and Interfaces
// =============================================================================

/** Parameter preset configuration */
export interface ParameterPreset {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly parameters: Record<string, unknown>;
  readonly createdAt: number;
  readonly version: string;
}

/** Parameter change history entry */
export interface ParameterHistoryEntry {
  readonly parameterId: string;
  readonly oldValue: unknown;
  readonly newValue: unknown;
  readonly timestamp: number;
}

/** Parameter change listener function */
export type ParameterChangeListener = (
  parameterId: string,
  newValue: unknown,
  oldValue: unknown
) => void;

/** Performance impact summary */
export interface PerformanceImpactSummary {
  readonly overall: PerformanceImpact;
  readonly breakdown: Record<PerformanceImpact, number>;
  readonly recommendations: readonly string[];
}

// =============================================================================
// Validation Functions
// =============================================================================

/** Validate radius parameters */
function validateRadius(value: number, type: 'top' | 'bottom'): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (type === 'top' && value < 1.0) {
    warnings.push({
      field: 'topRadius',
      message: 'Very small top radius may reduce exploration effectiveness',
      suggestion: 'Consider using a radius of at least 1.0'
    });
  }

  if (type === 'bottom' && value > 1.0) {
    warnings.push({
      field: 'bottomRadius',
      message: 'Large bottom radius may reduce focus effectiveness',
      suggestion: 'Consider using a radius less than 1.0 for better convergence'
    });
  }

  return { valid: errors.length === 0, errors, warnings };
}

/** Validate number of turns */
function validateTurns(value: unknown): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  const turns = value as number;

  if (turns > 50) {
    warnings.push({
      field: 'turns',
      message: 'High number of turns may impact visualization performance',
      suggestion: 'Consider reducing turns for better performance'
    });
  }

  if (turns < 5) {
    warnings.push({
      field: 'turns',
      message: 'Low number of turns may reduce helix effectiveness',
      suggestion: 'Consider using at least 5 turns for proper helix geometry'
    });
  }

  return { valid: errors.length === 0, errors, warnings };
}

/** Validate maximum agents */
function validateMaxAgents(value: unknown): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  const maxAgents = value as number;

  if (maxAgents > 50) {
    warnings.push({
      field: 'maxAgents',
      message: 'High agent count may cause memory issues',
      suggestion: 'Consider reducing max agents for better stability'
    });
  }

  return { valid: errors.length === 0, errors, warnings };
}

/** Validate agent types selection */
function validateAgentTypes(value: unknown): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  const types = value as AgentType[];

  if (types.length === 0) {
    errors.push({
      field: 'agentTypes',
      message: 'At least one agent type must be selected',
      code: 'NO_AGENTS'
    });
  }

  if (!types.includes(AgentType.Synthesis)) {
    warnings.push({
      field: 'agentTypes',
      message: 'Synthesis agents are recommended for task completion',
      suggestion: 'Consider including synthesis agents'
    });
  }

  return { valid: errors.length === 0, errors, warnings };
}

/** Validate GPU availability */
function validateGPUAvailability(value: unknown): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (value === true && !window.navigator.gpu) {
    warnings.push({
      field: 'enableGPU',
      message: 'GPU not available on this device',
      suggestion: 'GPU acceleration will be disabled automatically'
    });
  }

  return { valid: errors.length === 0, errors, warnings };
}

// Export for use in Gradio interface
export default ParameterControlManager;