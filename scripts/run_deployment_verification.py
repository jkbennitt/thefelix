#!/usr/bin/env python3
"""
Felix Framework - Deployment Verification Runner

Quick script to run deployment verification with different configurations.
This provides an easy way to verify deployment readiness across different
scenarios without running the full test suite.

Usage:
    python scripts/run_deployment_verification.py
    python scripts/run_deployment_verification.py --quick
    python scripts/run_deployment_verification.py --gpu-only
    python scripts/run_deployment_verification.py --export report.json
"""

import os
import sys
import asyncio
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deployment_verification import DeploymentVerificationFramework, setup_logging


async def quick_verification():
    """Run quick deployment verification (essential components only)."""
    print("🚀 Running Quick Felix Framework Deployment Verification...")

    framework = DeploymentVerificationFramework()

    # Run essential components only
    await framework._verify_core_mathematical_precision()
    await framework._verify_zerogpu_integration()
    await framework._verify_web_interface_compatibility()

    return framework._generate_deployment_report()


async def gpu_verification():
    """Run GPU-specific verification tests."""
    print("⚡ Running GPU-Specific Felix Framework Verification...")

    framework = DeploymentVerificationFramework()

    # Run GPU-related tests only
    await framework._verify_zerogpu_integration()
    await framework._verify_gpu_memory_management()

    return framework._generate_deployment_report()


def display_summary_report(report):
    """Display a concise summary of the verification report."""
    print("\n" + "="*60)
    print("🌪️  FELIX FRAMEWORK DEPLOYMENT SUMMARY")
    print("="*60)

    # Overall status
    status_emoji = "✅" if report.ready_for_deployment else "❌"
    print(f"Status: {status_emoji} {'READY FOR DEPLOYMENT' if report.ready_for_deployment else 'NOT READY'}")
    print(f"Overall Score: {report.overall_score:.1%}")

    # Component summary
    print("\n📊 Component Results:")
    components = {}
    for result in report.validation_results:
        if result.component not in components:
            components[result.component] = []
        components[result.component].append(result)

    for component, results in components.items():
        passed = len([r for r in results if r.success])
        total = len(results)
        avg_score = sum(r.score for r in results) / total
        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        print(f"  {status} {component}: {passed}/{total} tests passed ({avg_score:.1%})")

    # Critical issues
    if report.critical_issues:
        print("\n🚨 Critical Issues:")
        for issue in report.critical_issues[:3]:  # Top 3
            print(f"  - {issue}")

    # Top recommendations
    if report.recommendations:
        print("\n💡 Top Recommendations:")
        for rec in report.recommendations[:3]:  # Top 3
            print(f"  - {rec}")

    print(f"\n⏱️ Verification completed in {report.system_info.get('total_validation_time', 0):.1f} seconds")
    print("="*60)


async def main():
    """Main entry point for deployment verification runner."""
    parser = argparse.ArgumentParser(description='Felix Framework Deployment Verification Runner')
    parser.add_argument('--quick', action='store_true', help='Run quick verification (essential components)')
    parser.add_argument('--gpu-only', action='store_true', help='Run GPU-specific tests only')
    parser.add_argument('--full', action='store_true', help='Run full comprehensive verification')
    parser.add_argument('--export', help='Export detailed report to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Run appropriate verification
        if args.quick:
            report = await quick_verification()
        elif args.gpu_only:
            report = await gpu_verification()
        elif args.full:
            framework = DeploymentVerificationFramework()
            report = await framework.run_full_verification()
        else:
            # Default: run key components
            print("🔍 Running Standard Felix Framework Deployment Verification...")
            framework = DeploymentVerificationFramework()

            # Run core components
            await framework._verify_core_mathematical_precision()
            await framework._verify_zerogpu_integration()
            await framework._verify_web_interface_compatibility()
            await framework._verify_performance_benchmarks()

            report = framework._generate_deployment_report()

        # Display results
        display_summary_report(report)

        # Export if requested
        if args.export:
            import json
            with open(args.export, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\n📄 Detailed report exported to: {args.export}")

        # Return appropriate exit code
        return 0 if report.ready_for_deployment else 1

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        if args.verbose:
            import traceback
            print(traceback.format_exc())
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)