/**
 * Felix Framework - Educational Interface with Guided Tours
 *
 * Comprehensive educational system for Felix Framework with interactive
 * tutorials, step-by-step guided tours, contextual help, and progressive
 * learning paths for understanding helix-based multi-agent cognitive architecture.
 *
 * Features:
 * - Interactive step-by-step tutorials
 * - Contextual help and tooltips
 * - Progressive learning paths
 * - Visual demonstration overlays
 * - Knowledge checkpoints and quizzes
 * - Adaptive content based on user progress
 * - Mobile-optimized educational content
 * - Multi-language support preparation
 *
 * @version 1.0.0
 * @author Felix Framework Team
 */

import {
  EducationalContent,
  TourStep,
  TourAction,
  TourValidation,
  ContentType,
  DifficultyLevel,
  ContentResource
} from '../types/gradio-interface';

import {
  AgentType,
  AgentState,
  HelixGeometry
} from '../types/felix-core';

// =============================================================================
// Educational Content Types
// =============================================================================

/** Educational module configuration */
export interface EducationalModule {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly difficulty: DifficultyLevel;
  readonly estimatedMinutes: number;
  readonly prerequisites: readonly string[];
  readonly learningObjectives: readonly string[];
  readonly content: readonly EducationalContent[];
  readonly tour?: GuidedTour;
  readonly quiz?: QuizConfiguration;
  readonly resources: readonly ContentResource[];
}

/** Guided tour configuration */
export interface GuidedTour {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly steps: readonly TourStep[];
  readonly canSkip: boolean;
  readonly autoProgress: boolean;
  readonly completionCriteria: CompletionCriteria;
}

/** Quiz configuration */
export interface QuizConfiguration {
  readonly id: string;
  readonly title: string;
  readonly questions: readonly QuizQuestion[];
  readonly passingScore: number;
  readonly allowRetakes: boolean;
  readonly showCorrectAnswers: boolean;
}

/** Quiz question types */
export interface QuizQuestion {
  readonly id: string;
  readonly type: QuestionType;
  readonly question: string;
  readonly options?: readonly string[];
  readonly correctAnswer: string | readonly string[];
  readonly explanation?: string;
  readonly points: number;
}

/** Question types */
export enum QuestionType {
  MultipleChoice = 'multiple_choice',
  TrueFalse = 'true_false',
  ShortAnswer = 'short_answer',
  MultipleSelect = 'multiple_select',
  Ordering = 'ordering',
  Matching = 'matching'
}

/** Completion criteria */
export interface CompletionCriteria {
  readonly visitAllSteps: boolean;
  readonly completeActions: boolean;
  readonly passQuiz?: boolean;
  readonly timeSpentMinimum?: number;
  readonly customValidation?: (progress: TourProgress) => boolean;
}

/** Tour progress tracking */
export interface TourProgress {
  readonly tourId: string;
  readonly currentStepIndex: number;
  readonly completedSteps: Set<number>;
  readonly startTime: number;
  readonly timeSpentSeconds: number;
  readonly actionsCompleted: number;
  readonly validationsPassed: number;
  readonly isCompleted: boolean;
}

/** User learning profile */
export interface UserLearningProfile {
  readonly userId: string;
  readonly completedModules: Set<string>;
  readonly completedTours: Set<string>;
  readonly currentDifficulty: DifficultyLevel;
  readonly preferredPace: LearningPace;
  readonly strengths: readonly string[];
  readonly needsImprovement: readonly string[];
  readonly totalTimeSpent: number;
  readonly achievementsBadges: readonly string[];
}

/** Learning pace preferences */
export enum LearningPace {
  Slow = 'slow',
  Medium = 'medium',
  Fast = 'fast',
  SelfPaced = 'self_paced'
}

// =============================================================================
// Educational Content Definitions
// =============================================================================

/** Core Felix Framework educational modules */
export const FELIX_EDUCATIONAL_MODULES: readonly EducationalModule[] = [
  {
    id: 'felix_introduction',
    title: 'Introduction to Felix Framework',
    description: 'Learn the fundamentals of helix-based multi-agent cognitive architecture',
    difficulty: DifficultyLevel.Beginner,
    estimatedMinutes: 15,
    prerequisites: [],
    learningObjectives: [
      'Understand the concept of helix-based coordination',
      'Learn about agent specialization and roles',
      'Recognize the advantages over traditional architectures',
      'Identify key components of the Felix system'
    ],
    content: [
      {
        id: 'intro_overview',
        title: 'What is Felix Framework?',
        content: `
# Welcome to Felix Framework! 🌪️

Felix Framework revolutionizes multi-agent AI coordination through **helix-based cognitive architecture**.
Unlike traditional graph-based systems, Felix uses geometric spiral paths for natural agent convergence.

## Key Innovation: The Helix Model

The helix structure provides:
- **Natural Convergence**: Agents spiral from broad exploration to focused synthesis
- **Automatic Focusing**: Geometric tapering concentrates attention naturally
- **Efficient Communication**: O(N) spoke-based messaging vs O(N²) mesh complexity
- **Scalable Architecture**: Linear performance scaling up to 133+ agents

## Research Validation ✅

Felix has been rigorously tested with:
- 107+ passing unit tests
- Mathematical precision < 1e-12 error
- Statistical significance testing (H1 supported, p=0.0441)
- 75% memory efficiency improvement over mesh systems

Ready to explore this innovative approach to AI coordination?
        `,
        type: ContentType.Introduction,
        difficulty: DifficultyLevel.Beginner,
        estimatedTime: 5,
        prerequisites: [],
        resources: [
          {
            type: 'paper',
            title: 'Felix Framework Research Paper',
            url: 'https://github.com/CalebisGross/thefelix/blob/main/RESEARCH_LOG.md',
            description: 'Complete research documentation and validation'
          }
        ]
      }
    ],
    tour: {
      id: 'felix_intro_tour',
      title: 'Felix Framework Overview Tour',
      description: 'Interactive introduction to Felix Framework interface and concepts',
      canSkip: true,
      autoProgress: false,
      steps: [
        {
          id: 'welcome',
          target: 'body',
          title: 'Welcome to Felix Framework!',
          content: 'Let\'s explore the revolutionary helix-based multi-agent cognitive architecture. This tour will guide you through the key concepts and interface.',
          position: 'center',
          action: {
            type: 'highlight',
            duration: 2000
          }
        },
        {
          id: 'helix_visualization',
          target: '#helix-plot',
          title: 'The Helix Visualization',
          content: 'This 3D helix represents the core of Felix Framework. Agents spawn at the top (broad exploration) and spiral down to the bottom (focused synthesis). The geometry naturally concentrates attention with a 33,000x focusing ratio.',
          position: 'right',
          action: {
            type: 'click',
            value: 'camera_preset_default'
          },
          validation: {
            condition: (state) => true,
            message: 'Great! You can see the helix structure.',
            retry: false
          }
        },
        {
          id: 'agent_types',
          target: '#agent-selector',
          title: 'Agent Specialization',
          content: 'Felix uses four specialized agent types:\n🔍 Research (early spawn, high creativity)\n🧠 Analysis (mid-stage reasoning)\n🎨 Synthesis (late spawn, high precision)\n🔎 Critic (quality validation)',
          position: 'left'
        },
        {
          id: 'task_input',
          target: '#task-input',
          title: 'Task Processing',
          content: 'Enter tasks here for multi-agent processing. Agents coordinate through the helix structure, naturally converging from broad exploration to focused solutions.',
          position: 'bottom'
        }
      ],
      completionCriteria: {
        visitAllSteps: true,
        completeActions: true
      }
    },
    quiz: {
      id: 'felix_intro_quiz',
      title: 'Felix Framework Basics Quiz',
      passingScore: 80,
      allowRetakes: true,
      showCorrectAnswers: true,
      questions: [
        {
          id: 'q1',
          type: QuestionType.MultipleChoice,
          question: 'What is the main geometric structure used in Felix Framework?',
          options: ['Linear pipeline', 'Mesh network', 'Helix spiral', 'Tree hierarchy'],
          correctAnswer: 'Helix spiral',
          explanation: 'Felix uses a helix spiral structure for natural agent convergence from broad exploration to focused synthesis.',
          points: 25
        },
        {
          id: 'q2',
          type: QuestionType.TrueFalse,
          question: 'Felix Framework has O(N²) communication complexity like traditional mesh systems.',
          correctAnswer: 'false',
          explanation: 'Felix uses O(N) spoke-based communication, which is more efficient than O(N²) mesh systems.',
          points: 25
        },
        {
          id: 'q3',
          type: QuestionType.MultipleSelect,
          question: 'Which agent types are available in Felix Framework? (Select all that apply)',
          options: ['Research', 'Analysis', 'Synthesis', 'Critic', 'Coordinator', 'Monitor'],
          correctAnswer: ['Research', 'Analysis', 'Synthesis', 'Critic'],
          explanation: 'Felix has four specialized agent types: Research, Analysis, Synthesis, and Critic.',
          points: 25
        },
        {
          id: 'q4',
          type: QuestionType.MultipleChoice,
          question: 'What is the concentration ratio of the Felix helix?',
          options: ['1,000x', '10,000x', '33,000x', '100,000x'],
          correctAnswer: '33,000x',
          explanation: 'The Felix helix has a 33,000x concentration ratio (33.0 top radius / 0.001 bottom radius).',
          points: 25
        }
      ]
    },
    resources: [
      {
        type: 'link',
        title: 'GitHub Repository',
        url: 'https://github.com/CalebisGross/thefelix',
        description: 'Complete source code and documentation'
      },
      {
        type: 'paper',
        title: 'Mathematical Model Documentation',
        url: 'https://github.com/CalebisGross/thefelix/blob/main/docs/architecture/core/mathematical_model.md',
        description: 'Detailed mathematical foundations'
      }
    ]
  },

  {
    id: 'helix_geometry',
    title: 'Understanding Helix Geometry',
    description: 'Deep dive into the mathematical foundations of the Felix helix',
    difficulty: DifficultyLevel.Intermediate,
    estimatedMinutes: 20,
    prerequisites: ['felix_introduction'],
    learningObjectives: [
      'Understand parametric helix equations',
      'Learn about concentration ratios and focusing',
      'Explore agent positioning mathematics',
      'Understand precision and validation requirements'
    ],
    content: [
      {
        id: 'helix_math',
        title: 'Mathematical Foundation',
        content: `
# Helix Mathematics 📐

The Felix helix is defined by precise parametric equations that create a natural focusing mechanism.

## Parametric Equations

**Position Vector**: r(t) = (R(t)cos(θ(t)), R(t)sin(θ(t)), Ht)

Where:
- **t ∈ [0,1]**: Parameter where t=0 is top, t=1 is bottom
- **R(t) = R_bottom × (R_top/R_bottom)^t**: Radius function with exponential tapering
- **θ(t) = 2πnt**: Angular function with n=33 complete turns
- **H = 100**: Total height

## Key Parameters

- **Top Radius**: 33.0 (broad exploration phase)
- **Bottom Radius**: 0.001 (focused synthesis phase)
- **Turns**: 33 complete rotations
- **Height**: 100.0 units
- **Concentration Ratio**: 33,000x focusing power

## Mathematical Precision

Felix maintains mathematical precision of **<1e-12 error** against the OpenSCAD prototype, ensuring research-grade accuracy for cognitive architecture modeling.

## Agent Positioning

Agents spawn at different t values along the helix:
- Research agents: t ≈ 0.1 (near top, high creativity)
- Analysis agents: t ≈ 0.5 (middle, balanced reasoning)
- Synthesis agents: t ≈ 0.9 (near bottom, high precision)

This positioning naturally implements temperature gradients for LLM creativity control.
        `,
        type: ContentType.Tutorial,
        difficulty: DifficultyLevel.Intermediate,
        estimatedTime: 10,
        prerequisites: ['felix_introduction'],
        resources: []
      }
    ],
    tour: {
      id: 'helix_geometry_tour',
      title: 'Exploring Helix Mathematics',
      description: 'Interactive exploration of helix geometry and mathematical properties',
      canSkip: true,
      autoProgress: false,
      steps: [
        {
          id: 'helix_parameters',
          target: '#helix-controls',
          title: 'Helix Parameter Controls',
          content: 'These controls let you modify the helix geometry. Try changing the top radius to see how it affects the concentration ratio and agent positioning.',
          position: 'right',
          action: {
            type: 'input',
            value: '25.0'
          }
        },
        {
          id: 'concentration_demo',
          target: '#concentration-indicator',
          title: 'Concentration Visualization',
          content: 'The red dashed circles show the dramatic size difference between top and bottom. This 33,000x concentration creates natural attention focusing.',
          position: 'top'
        },
        {
          id: 'agent_positioning',
          target: '#agent-positions',
          title: 'Agent Positioning Math',
          content: 'Watch how agents are positioned along the helix using the t parameter. Each agent type has optimal t values for their cognitive role.',
          position: 'bottom'
        }
      ],
      completionCriteria: {
        visitAllSteps: true,
        completeActions: true
      }
    },
    resources: [
      {
        type: 'file',
        title: 'OpenSCAD Prototype',
        url: 'https://github.com/CalebisGross/thefelix/blob/main/thefelix.md',
        description: 'Original 3D model demonstrating core concepts'
      }
    ]
  },

  {
    id: 'agent_coordination',
    title: 'Multi-Agent Coordination',
    description: 'Learn how agents coordinate through the helix structure',
    difficulty: DifficultyLevel.Intermediate,
    estimatedMinutes: 25,
    prerequisites: ['felix_introduction', 'helix_geometry'],
    learningObjectives: [
      'Understand spoke-based communication',
      'Learn agent lifecycle management',
      'Explore task distribution patterns',
      'Compare with traditional architectures'
    ],
    content: [
      {
        id: 'coordination_overview',
        title: 'Agent Coordination Patterns',
        content: `
# Multi-Agent Coordination in Felix 🤝

Felix implements a novel coordination pattern that leverages geometric properties for efficient agent communication and task distribution.

## Spoke-Based Communication

Unlike traditional mesh networks with O(N²) complexity, Felix uses **spoke-based communication**:

- **Central Post**: Hub for all coordination
- **Agent Spokes**: Direct O(N) connections to center
- **No Agent-to-Agent**: Eliminates complex mesh routing
- **Natural Load Balancing**: Geometric distribution handles traffic

## Agent Lifecycle

1. **Spawning**: Agents spawn at helix top at different times
2. **Traversal**: Natural progression down the helix spiral
3. **Specialization**: Temperature/creativity decreases with depth
4. **Coordination**: Spoke-based messaging throughout lifecycle
5. **Completion**: Natural convergence at bottom focus point

## Task Distribution Efficiency

Research validation shows **statistically significant improvement** (p=0.0441) in task distribution efficiency compared to linear pipelines:

- **H1 SUPPORTED**: Better task distribution through geometric properties
- **Memory Efficiency**: 75% reduction vs mesh systems (1,200 vs 4,800 units)
- **Scalability**: Linear performance up to 133+ agents

## Comparison with Traditional Architectures

| Feature | Felix Helix | Linear Pipeline | Mesh Network |
|---------|-------------|-----------------|--------------|
| Communication | O(N) spokes | O(N×M) stages | O(N²) mesh |
| Memory Usage | 1,200 units | 2,400 units | 4,800 units |
| Scalability | Linear | Limited stages | Quadratic |
| Natural Focus | Geometric | Manual | None |
| Setup Complexity | Low | Medium | High |
        `,
        type: ContentType.Tutorial,
        difficulty: DifficultyLevel.Intermediate,
        estimatedTime: 15,
        prerequisites: ['felix_introduction'],
        resources: []
      }
    ],
    tour: {
      id: 'coordination_tour',
      title: 'Agent Coordination Demo',
      description: 'See multi-agent coordination in action',
      canSkip: false,
      autoProgress: true,
      steps: [
        {
          id: 'start_task',
          target: '#process-button',
          title: 'Start Multi-Agent Processing',
          content: 'Click to start a task and watch how agents coordinate through the helix structure.',
          position: 'top',
          action: {
            type: 'click'
          },
          validation: {
            condition: (state) => (state as any).taskStarted === true,
            message: 'Task started! Watch the agent coordination.',
            retry: true
          }
        },
        {
          id: 'observe_spawning',
          target: '#helix-plot',
          title: 'Agent Spawning Pattern',
          content: 'Notice how agents appear at different helix positions. Research agents spawn early (top), synthesis agents spawn late (bottom).',
          position: 'right'
        },
        {
          id: 'spoke_communication',
          target: '#communication-indicator',
          title: 'Spoke Communication',
          content: 'The white dotted lines show spoke-based communication between agents and the central post. This O(N) pattern is much more efficient than mesh networks.',
          position: 'left'
        }
      ],
      completionCriteria: {
        visitAllSteps: true,
        completeActions: true,
        customValidation: (progress) => progress.actionsCompleted >= 1
      }
    },
    resources: [
      {
        type: 'paper',
        title: 'Statistical Validation Results',
        url: 'https://github.com/CalebisGross/thefelix/blob/main/docs/architecture/core/hypothesis_mathematics.md',
        description: 'Complete statistical analysis and hypothesis testing'
      }
    ]
  },

  {
    id: 'performance_optimization',
    title: 'Performance Optimization',
    description: 'Advanced techniques for optimizing Felix Framework performance',
    difficulty: DifficultyLevel.Advanced,
    estimatedMinutes: 30,
    prerequisites: ['felix_introduction', 'helix_geometry', 'agent_coordination'],
    learningObjectives: [
      'Understand performance bottlenecks',
      'Learn optimization strategies',
      'Configure for different environments',
      'Monitor and analyze performance metrics'
    ],
    content: [
      {
        id: 'performance_fundamentals',
        title: 'Performance Optimization Strategies',
        content: `
# Performance Optimization 🚀

Felix Framework provides multiple optimization strategies for different deployment scenarios and performance requirements.

## Key Performance Factors

### 1. Helix Resolution
- **High Resolution (1000+ points)**: Better visual quality, higher GPU/CPU load
- **Medium Resolution (500 points)**: Balanced quality/performance (default)
- **Low Resolution (100-200 points)**: Maximum performance, suitable for mobile

### 2. Agent Management
- **Max Agents**: Higher counts increase memory usage exponentially
- **Spawn Timing**: Staggered spawning reduces instantaneous load
- **Lifecycle Management**: Proper cleanup prevents memory leaks

### 3. Communication Optimization
- **Spoke Efficiency**: O(N) complexity scales linearly
- **Message Batching**: Group updates for better throughput
- **Compression**: Reduce bandwidth for mobile/slow connections

### 4. Rendering Optimization
- **WebGL Acceleration**: Use GPU for 3D rendering when available
- **Level of Detail (LOD)**: Reduce complexity based on camera distance
- **Frustum Culling**: Only render visible portions
- **Animation Throttling**: Reduce frame rate on low-power devices

## Environment-Specific Optimizations

### Mobile Devices
```typescript
{
  helixResolution: 150,
  updateInterval: 2000,
  renderingMode: 'mobile',
  compressionLevel: 6,
  enableAnimations: false
}
```

### High-Performance Desktop
```typescript
{
  helixResolution: 1000,
  updateInterval: 250,
  renderingMode: 'quality',
  maxAgents: 50,
  enableGPU: true
}
```

### Cloud/Server Deployment
```typescript
{
  memoryLimit: 2000,
  compressionLevel: 3,
  batchUpdates: true,
  cacheSize: 200
}
```

## Performance Monitoring

Felix provides comprehensive performance metrics:
- **Response Time**: Agent coordination latency
- **Memory Usage**: Heap allocation and GC pressure
- **GPU Utilization**: Rendering performance
- **Network Throughput**: Communication efficiency
- **Error Rates**: System stability indicators

## Optimization Best Practices

1. **Start with Balanced Settings**: Use default configuration first
2. **Profile Before Optimizing**: Identify actual bottlenecks
3. **Test on Target Hardware**: Mobile performance varies significantly
4. **Monitor Continuously**: Performance can degrade over time
5. **Use Performance Presets**: Apply tested configurations for common scenarios
        `,
        type: ContentType.Tutorial,
        difficulty: DifficultyLevel.Advanced,
        estimatedTime: 20,
        prerequisites: ['agent_coordination'],
        resources: []
      }
    ],
    resources: [
      {
        type: 'link',
        title: 'Performance Benchmarking Guide',
        url: 'https://github.com/CalebisGross/thefelix/blob/main/benchmarks/',
        description: 'Detailed benchmarking tools and results'
      }
    ]
  }
];

// =============================================================================
// Educational Interface Manager
// =============================================================================

export class EducationalInterfaceManager {
  private currentModule: EducationalModule | null = null;
  private currentTour: GuidedTour | null = null;
  private tourProgress: TourProgress | null = null;
  private userProfile: UserLearningProfile;
  private helpOverlay: HTMLElement | null = null;
  private tourOverlay: HTMLElement | null = null;

  constructor(userId: string = 'default_user') {
    this.userProfile = this.initializeUserProfile(userId);
    this.setupEventListeners();
  }

  // =============================================================================
  // Module Management
  // =============================================================================

  /**
   * Get available educational modules
   */
  public getAvailableModules(): readonly EducationalModule[] {
    return FELIX_EDUCATIONAL_MODULES.filter(module =>
      this.hasPrerequisites(module.prerequisites)
    );
  }

  /**
   * Start an educational module
   */
  public startModule(moduleId: string): boolean {
    const module = FELIX_EDUCATIONAL_MODULES.find(m => m.id === moduleId);
    if (!module) return false;

    if (!this.hasPrerequisites(module.prerequisites)) {
      console.warn(`Missing prerequisites for module ${moduleId}`);
      return false;
    }

    this.currentModule = module;
    return true;
  }

  /**
   * Complete current module
   */
  public completeModule(): void {
    if (this.currentModule) {
      this.userProfile.completedModules.add(this.currentModule.id);
      this.updateUserProgress();
      this.currentModule = null;
    }
  }

  /**
   * Check if user has required prerequisites
   */
  private hasPrerequisites(prerequisites: readonly string[]): boolean {
    return prerequisites.every(prereq =>
      this.userProfile.completedModules.has(prereq)
    );
  }

  // =============================================================================
  // Guided Tour Management
  // =============================================================================

  /**
   * Start a guided tour
   */
  public startTour(tourId: string): boolean {
    const module = FELIX_EDUCATIONAL_MODULES.find(m => m.tour?.id === tourId);
    if (!module?.tour) return false;

    this.currentTour = module.tour;
    this.tourProgress = {
      tourId,
      currentStepIndex: 0,
      completedSteps: new Set(),
      startTime: Date.now(),
      timeSpentSeconds: 0,
      actionsCompleted: 0,
      validationsPassed: 0,
      isCompleted: false
    };

    this.showTourStep(0);
    return true;
  }

  /**
   * Navigate to next tour step
   */
  public nextTourStep(): boolean {
    if (!this.currentTour || !this.tourProgress) return false;

    const currentStep = this.currentTour.steps[this.tourProgress.currentStepIndex];

    // Validate current step if required
    if (currentStep.validation && !this.validateTourStep(currentStep)) {
      return false;
    }

    // Mark current step as completed
    this.tourProgress.completedSteps.add(this.tourProgress.currentStepIndex);

    // Move to next step
    if (this.tourProgress.currentStepIndex < this.currentTour.steps.length - 1) {
      this.tourProgress.currentStepIndex++;
      this.showTourStep(this.tourProgress.currentStepIndex);
      return true;
    } else {
      this.completeTour();
      return false;
    }
  }

  /**
   * Navigate to previous tour step
   */
  public previousTourStep(): boolean {
    if (!this.tourProgress || this.tourProgress.currentStepIndex === 0) return false;

    this.tourProgress.currentStepIndex--;
    this.showTourStep(this.tourProgress.currentStepIndex);
    return true;
  }

  /**
   * Skip current tour
   */
  public skipTour(): void {
    if (!this.currentTour || !this.currentTour.canSkip) return;

    this.hideTourOverlay();
    this.currentTour = null;
    this.tourProgress = null;
  }

  /**
   * Complete current tour
   */
  private completeTour(): void {
    if (!this.currentTour || !this.tourProgress) return;

    // Check completion criteria
    const criteria = this.currentTour.completionCriteria;
    const meetsVisitCriteria = !criteria.visitAllSteps ||
      this.tourProgress.completedSteps.size === this.currentTour.steps.length;
    const meetsActionCriteria = !criteria.completeActions ||
      this.tourProgress.actionsCompleted > 0;
    const meetsCustomCriteria = !criteria.customValidation ||
      criteria.customValidation(this.tourProgress);

    if (meetsVisitCriteria && meetsActionCriteria && meetsCustomCriteria) {
      this.tourProgress.isCompleted = true;
      this.userProfile.completedTours.add(this.currentTour.id);
      this.updateUserProgress();
      this.showTourCompletion();
    }

    this.hideTourOverlay();
    this.currentTour = null;
    this.tourProgress = null;
  }

  /**
   * Show current tour step
   */
  private showTourStep(stepIndex: number): void {
    if (!this.currentTour) return;

    const step = this.currentTour.steps[stepIndex];
    const targetElement = this.findTargetElement(step.target);

    if (!targetElement && step.target !== 'body') {
      console.warn(`Tour step target not found: ${step.target}`);
      return;
    }

    // Create or update tour overlay
    this.updateTourOverlay(step, stepIndex);

    // Perform step action if specified
    if (step.action) {
      this.performTourAction(step.action, targetElement);
    }

    // Highlight target element
    if (targetElement && step.target !== 'body') {
      this.highlightElement(targetElement);
    }
  }

  /**
   * Validate tour step completion
   */
  private validateTourStep(step: TourStep): boolean {
    if (!step.validation || !this.tourProgress) return true;

    const isValid = step.validation.condition(this.getTourState());

    if (!isValid && step.validation.retry) {
      // Show validation message and allow retry
      this.showValidationMessage(step.validation.message);
      return false;
    }

    if (isValid) {
      this.tourProgress.validationsPassed++;
    }

    return isValid;
  }

  /**
   * Perform tour action
   */
  private performTourAction(action: TourAction, targetElement: Element | null): void {
    if (!this.tourProgress) return;

    switch (action.type) {
      case 'click':
        if (targetElement) {
          (targetElement as HTMLElement).click();
          this.tourProgress.actionsCompleted++;
        }
        break;
      case 'input':
        if (targetElement && action.value) {
          (targetElement as HTMLInputElement).value = action.value;
          targetElement.dispatchEvent(new Event('input', { bubbles: true }));
          this.tourProgress.actionsCompleted++;
        }
        break;
      case 'highlight':
        if (targetElement) {
          this.highlightElement(targetElement, action.duration);
        }
        break;
      case 'wait':
        if (action.duration) {
          setTimeout(() => {
            if (this.currentTour?.autoProgress) {
              this.nextTourStep();
            }
          }, action.duration);
        }
        break;
    }
  }

  // =============================================================================
  // Contextual Help System
  // =============================================================================

  /**
   * Show contextual help for element
   */
  public showContextualHelp(elementId: string, content: string): void {
    const element = document.getElementById(elementId);
    if (!element) return;

    this.createHelpTooltip(element, content);
  }

  /**
   * Create help tooltip
   */
  private createHelpTooltip(element: Element, content: string): void {
    // Remove existing tooltips
    this.removeExistingTooltips();

    const tooltip = document.createElement('div');
    tooltip.className = 'felix-help-tooltip';
    tooltip.innerHTML = `
      <div class="tooltip-content">
        <button class="tooltip-close">&times;</button>
        <div class="tooltip-text">${content}</div>
      </div>
    `;

    // Position tooltip
    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'absolute';
    tooltip.style.top = `${rect.bottom + 10}px`;
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.zIndex = '10000';

    // Add to page
    document.body.appendChild(tooltip);

    // Add close handler
    const closeButton = tooltip.querySelector('.tooltip-close');
    closeButton?.addEventListener('click', () => {
      tooltip.remove();
    });

    // Auto-remove after delay
    setTimeout(() => {
      if (tooltip.parentNode) {
        tooltip.remove();
      }
    }, 10000);
  }

  /**
   * Remove existing tooltips
   */
  private removeExistingTooltips(): void {
    const tooltips = document.querySelectorAll('.felix-help-tooltip');
    tooltips.forEach(tooltip => tooltip.remove());
  }

  // =============================================================================
  // Quiz System
  // =============================================================================

  /**
   * Start quiz for current module
   */
  public startQuiz(): QuizSession | null {
    if (!this.currentModule?.quiz) return null;

    return new QuizSession(this.currentModule.quiz, this.userProfile);
  }

  // =============================================================================
  // User Interface Management
  // =============================================================================

  /**
   * Update tour overlay
   */
  private updateTourOverlay(step: TourStep, stepIndex: number): void {
    if (!this.currentTour || !this.tourProgress) return;

    // Remove existing overlay
    this.hideTourOverlay();

    // Create new overlay
    this.tourOverlay = document.createElement('div');
    this.tourOverlay.className = 'felix-tour-overlay';
    this.tourOverlay.innerHTML = `
      <div class="tour-backdrop"></div>
      <div class="tour-popup ${step.position}">
        <div class="tour-header">
          <h3>${step.title}</h3>
          <div class="tour-progress">
            ${stepIndex + 1} of ${this.currentTour.steps.length}
          </div>
        </div>
        <div class="tour-content">
          ${step.content}
        </div>
        <div class="tour-controls">
          <button class="tour-btn tour-prev" ${stepIndex === 0 ? 'disabled' : ''}>
            Previous
          </button>
          <button class="tour-btn tour-skip" ${!this.currentTour.canSkip ? 'style="display:none"' : ''}>
            Skip Tour
          </button>
          <button class="tour-btn tour-next primary">
            ${stepIndex === this.currentTour.steps.length - 1 ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    `;

    // Position popup relative to target
    if (step.target !== 'body') {
      const targetElement = this.findTargetElement(step.target);
      if (targetElement) {
        this.positionTourPopup(targetElement, step.position);
      }
    }

    // Add event listeners
    this.setupTourEventListeners();

    // Add to page
    document.body.appendChild(this.tourOverlay);
  }

  /**
   * Hide tour overlay
   */
  private hideTourOverlay(): void {
    if (this.tourOverlay) {
      this.tourOverlay.remove();
      this.tourOverlay = null;
    }

    // Remove highlights
    this.removeHighlights();
  }

  /**
   * Position tour popup relative to target
   */
  private positionTourPopup(targetElement: Element, position: string): void {
    if (!this.tourOverlay) return;

    const popup = this.tourOverlay.querySelector('.tour-popup') as HTMLElement;
    const rect = targetElement.getBoundingClientRect();

    switch (position) {
      case 'top':
        popup.style.bottom = `${window.innerHeight - rect.top + 10}px`;
        popup.style.left = `${rect.left + rect.width / 2}px`;
        popup.style.transform = 'translateX(-50%)';
        break;
      case 'bottom':
        popup.style.top = `${rect.bottom + 10}px`;
        popup.style.left = `${rect.left + rect.width / 2}px`;
        popup.style.transform = 'translateX(-50%)';
        break;
      case 'left':
        popup.style.top = `${rect.top + rect.height / 2}px`;
        popup.style.right = `${window.innerWidth - rect.left + 10}px`;
        popup.style.transform = 'translateY(-50%)';
        break;
      case 'right':
        popup.style.top = `${rect.top + rect.height / 2}px`;
        popup.style.left = `${rect.right + 10}px`;
        popup.style.transform = 'translateY(-50%)';
        break;
      case 'center':
        popup.style.top = '50%';
        popup.style.left = '50%';
        popup.style.transform = 'translate(-50%, -50%)';
        break;
    }
  }

  /**
   * Setup tour event listeners
   */
  private setupTourEventListeners(): void {
    if (!this.tourOverlay) return;

    const prevBtn = this.tourOverlay.querySelector('.tour-prev');
    const nextBtn = this.tourOverlay.querySelector('.tour-next');
    const skipBtn = this.tourOverlay.querySelector('.tour-skip');

    prevBtn?.addEventListener('click', () => this.previousTourStep());
    nextBtn?.addEventListener('click', () => this.nextTourStep());
    skipBtn?.addEventListener('click', () => this.skipTour());

    // Keyboard navigation
    document.addEventListener('keydown', this.handleTourKeydown.bind(this));
  }

  /**
   * Handle tour keyboard navigation
   */
  private handleTourKeydown(event: KeyboardEvent): void {
    if (!this.currentTour) return;

    switch (event.key) {
      case 'ArrowRight':
      case 'Enter':
        event.preventDefault();
        this.nextTourStep();
        break;
      case 'ArrowLeft':
        event.preventDefault();
        this.previousTourStep();
        break;
      case 'Escape':
        event.preventDefault();
        if (this.currentTour.canSkip) {
          this.skipTour();
        }
        break;
    }
  }

  /**
   * Highlight target element
   */
  private highlightElement(element: Element, duration?: number): void {
    element.classList.add('felix-tour-highlight');

    if (duration) {
      setTimeout(() => {
        element.classList.remove('felix-tour-highlight');
      }, duration);
    }
  }

  /**
   * Remove all highlights
   */
  private removeHighlights(): void {
    const highlights = document.querySelectorAll('.felix-tour-highlight');
    highlights.forEach(el => el.classList.remove('felix-tour-highlight'));
  }

  /**
   * Show tour completion message
   */
  private showTourCompletion(): void {
    if (!this.currentTour) return;

    const completion = document.createElement('div');
    completion.className = 'felix-tour-completion';
    completion.innerHTML = `
      <div class="completion-content">
        <h2>🎉 Tour Completed!</h2>
        <p>You've successfully completed the ${this.currentTour.title} tour.</p>
        <button class="completion-close">Continue</button>
      </div>
    `;

    document.body.appendChild(completion);

    const closeBtn = completion.querySelector('.completion-close');
    closeBtn?.addEventListener('click', () => {
      completion.remove();
    });

    setTimeout(() => {
      if (completion.parentNode) {
        completion.remove();
      }
    }, 5000);
  }

  /**
   * Show validation message
   */
  private showValidationMessage(message: string): void {
    const validation = document.createElement('div');
    validation.className = 'felix-tour-validation';
    validation.innerHTML = `
      <div class="validation-content">
        <p>${message}</p>
      </div>
    `;

    document.body.appendChild(validation);

    setTimeout(() => {
      validation.remove();
    }, 3000);
  }

  // =============================================================================
  // Utility Methods
  // =============================================================================

  /**
   * Find target element for tour step
   */
  private findTargetElement(selector: string): Element | null {
    if (selector === 'body') return document.body;

    // Try ID first, then querySelector
    return document.getElementById(selector.replace('#', '')) ||
           document.querySelector(selector);
  }

  /**
   * Get current tour state for validation
   */
  private getTourState(): Record<string, unknown> {
    return {
      tourProgress: this.tourProgress,
      userProfile: this.userProfile,
      currentModule: this.currentModule,
      // Add more state as needed for validation
    };
  }

  /**
   * Initialize user learning profile
   */
  private initializeUserProfile(userId: string): UserLearningProfile {
    // In real implementation, this would load from storage
    return {
      userId,
      completedModules: new Set(),
      completedTours: new Set(),
      currentDifficulty: DifficultyLevel.Beginner,
      preferredPace: LearningPace.Medium,
      strengths: [],
      needsImprovement: [],
      totalTimeSpent: 0,
      achievementsBadges: []
    };
  }

  /**
   * Update user progress
   */
  private updateUserProgress(): void {
    // In real implementation, this would save to storage/database
    console.log('User progress updated:', this.userProfile);
  }

  /**
   * Setup global event listeners
   */
  private setupEventListeners(): void {
    // Help button clicks
    document.addEventListener('click', (event) => {
      const target = event.target as HTMLElement;
      if (target.classList.contains('felix-help-trigger')) {
        const helpContent = target.dataset.helpContent;
        if (helpContent) {
          this.showContextualHelp(target.id, helpContent);
        }
      }
    });
  }

  // =============================================================================
  // Public API
  // =============================================================================

  /**
   * Get user learning progress
   */
  public getUserProgress(): UserLearningProfile {
    return { ...this.userProfile };
  }

  /**
   * Get recommended next module
   */
  public getRecommendedModule(): EducationalModule | null {
    const available = this.getAvailableModules();
    const incomplete = available.filter(module =>
      !this.userProfile.completedModules.has(module.id)
    );

    if (incomplete.length === 0) return null;

    // Sort by difficulty and prerequisites
    incomplete.sort((a, b) => {
      const difficultyOrder = {
        [DifficultyLevel.Beginner]: 0,
        [DifficultyLevel.Intermediate]: 1,
        [DifficultyLevel.Advanced]: 2,
        [DifficultyLevel.Expert]: 3
      };

      return difficultyOrder[a.difficulty] - difficultyOrder[b.difficulty] ||
             a.prerequisites.length - b.prerequisites.length;
    });

    return incomplete[0];
  }

  /**
   * Dispose of resources
   */
  public dispose(): void {
    this.hideTourOverlay();
    this.removeExistingTooltips();
    document.removeEventListener('keydown', this.handleTourKeydown);
  }
}

// =============================================================================
// Quiz Session Class
// =============================================================================

export class QuizSession {
  private currentQuestionIndex: number = 0;
  private answers: Map<string, string | string[]> = new Map();
  private score: number = 0;
  private startTime: number = Date.now();

  constructor(
    private readonly quiz: QuizConfiguration,
    private readonly userProfile: UserLearningProfile
  ) {}

  public getCurrentQuestion(): QuizQuestion | null {
    if (this.currentQuestionIndex >= this.quiz.questions.length) return null;
    return this.quiz.questions[this.currentQuestionIndex];
  }

  public submitAnswer(answer: string | string[]): void {
    const currentQuestion = this.getCurrentQuestion();
    if (!currentQuestion) return;

    this.answers.set(currentQuestion.id, answer);
  }

  public nextQuestion(): boolean {
    if (this.currentQuestionIndex < this.quiz.questions.length - 1) {
      this.currentQuestionIndex++;
      return true;
    }
    return false;
  }

  public previousQuestion(): boolean {
    if (this.currentQuestionIndex > 0) {
      this.currentQuestionIndex--;
      return true;
    }
    return false;
  }

  public finishQuiz(): QuizResult {
    this.calculateScore();

    const result: QuizResult = {
      quizId: this.quiz.id,
      score: this.score,
      totalPoints: this.quiz.questions.reduce((sum, q) => sum + q.points, 0),
      passed: this.score >= this.quiz.passingScore,
      timeSpent: Date.now() - this.startTime,
      answers: new Map(this.answers),
      feedback: this.generateFeedback()
    };

    return result;
  }

  private calculateScore(): void {
    this.score = 0;

    this.quiz.questions.forEach(question => {
      const userAnswer = this.answers.get(question.id);
      if (!userAnswer) return;

      const isCorrect = Array.isArray(question.correctAnswer)
        ? this.arraysEqual(userAnswer as string[], question.correctAnswer)
        : userAnswer === question.correctAnswer;

      if (isCorrect) {
        this.score += question.points;
      }
    });
  }

  private arraysEqual(a: string[], b: string[]): boolean {
    if (a.length !== b.length) return false;
    const sortedA = [...a].sort();
    const sortedB = [...b].sort();
    return sortedA.every((val, index) => val === sortedB[index]);
  }

  private generateFeedback(): string[] {
    const feedback: string[] = [];

    if (this.score >= this.quiz.passingScore) {
      feedback.push('Congratulations! You passed the quiz.');
    } else {
      feedback.push('You did not meet the passing score. Consider reviewing the material.');
    }

    // Add specific feedback based on incorrect answers
    this.quiz.questions.forEach(question => {
      const userAnswer = this.answers.get(question.id);
      const isCorrect = Array.isArray(question.correctAnswer)
        ? this.arraysEqual(userAnswer as string[], question.correctAnswer)
        : userAnswer === question.correctAnswer;

      if (!isCorrect && question.explanation) {
        feedback.push(`Question "${question.question}": ${question.explanation}`);
      }
    });

    return feedback;
  }
}

// =============================================================================
// Supporting Types
// =============================================================================

export interface QuizResult {
  readonly quizId: string;
  readonly score: number;
  readonly totalPoints: number;
  readonly passed: boolean;
  readonly timeSpent: number;
  readonly answers: Map<string, string | string[]>;
  readonly feedback: readonly string[];
}

// Export for use in Gradio interface
export default EducationalInterfaceManager;