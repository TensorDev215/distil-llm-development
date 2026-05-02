#!/usr/bin/env python3
"""
Compute composite scores from eval metrics.

This script takes evaluation results (from pod_eval_vllm.py) and computes
composite scores for each model using the multi-axis scoring logic.

Usage:
    python3 compute_composite_score.py \\
        --results results.json \\
        --output results_with_composite.json \\
        --king-kl 0.5 \\
        --king-rkl 0.6
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Add scripts to path to import validator modules
sys.path.insert(0, str(Path(__file__).parent))

from validator.composite import (
    compute_composite,
    compute_axes,
    resolve_reference_broken_axes,
    resolve_teacher_broken_axes,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_results(results_path: str) -> dict:
    """Load evaluation results from JSON file.
    
    Args:
        results_path: Path to the results JSON file from pod_eval_vllm.py
        
    Returns:
        Dictionary containing the evaluation results
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    logger.info(f"Loading results from {results_path}")
    with open(path, "r") as f:
        data = json.load(f)
    
    logger.info(f"Loaded results for {len(data.get('students', {}))} models")
    return data


def extract_king_values(results: dict, king_model: Optional[str] = None) -> tuple:
    """Extract king's KL and RKL values.
    
    Args:
        results: The evaluation results dictionary
        king_model: Model name to use as king. If None, uses the first model.
        
    Returns:
        Tuple of (king_kl, king_rkl, king_eopd, king_kl_is, king_forking_rkl, 
                  king_trace_nll, king_kl_tail)
    """
    students = results.get("students", {})
    
    if king_model:
        if king_model not in students:
            logger.warning(f"King model {king_model} not found in results")
            king_data = next(iter(students.values())) if students else {}
        else:
            king_data = students[king_model]
    else:
        # Use first model as king if not specified
        king_data = next(iter(students.values())) if students else {}
    
    king_kl = king_data.get("kl_global_avg")
    king_rkl = king_data.get("on_policy_rkl", {}).get("mean")
    king_eopd = king_data.get("eopd_adaptive_mean")
    king_kl_is = king_data.get("kl_is_mean")
    king_forking_rkl = king_data.get("forking_rkl_mean")
    king_trace_nll = king_data.get("teacher_trace_nll_mean")
    king_kl_tail = king_data.get("kl_tail_mean")
    
    logger.info(
        f"King metrics: kl={king_kl}, rkl={king_rkl}, eopd={king_eopd}, "
        f"kl_is={king_kl_is}, forking_rkl={king_forking_rkl}"
    )
    
    return king_kl, king_rkl, king_eopd, king_kl_is, king_forking_rkl, king_trace_nll, king_kl_tail


def compute_composites_for_all(
    results: dict,
    king_kl: Optional[float] = None,
    king_rkl: Optional[float] = None,
    king_eopd: Optional[float] = None,
    king_kl_is: Optional[float] = None,
    king_forking_rkl: Optional[float] = None,
    king_trace_nll: Optional[float] = None,
    king_kl_tail: Optional[float] = None,
) -> dict:
    """Compute composite scores for all models in results.
    
    Args:
        results: The evaluation results dictionary
        king_kl: King's KL value for reference. If None, auto-extract.
        king_rkl: King's RKL value for reference. If None, auto-extract.
        king_eopd: King's EOPD value for reference.
        king_kl_is: King's KL-IS value for reference.
        king_forking_rkl: King's forking RKL value for reference.
        king_trace_nll: King's teacher trace NLL value for reference.
        king_kl_tail: King's KL tail value for reference.
        
    Returns:
        Dictionary with composite scores added to each student
    """
    students = results.get("students", {})
    
    # If king values not provided, extract from results
    if king_kl is None or king_rkl is None:
        k_kl, k_rkl, k_eopd, k_kl_is, k_frk, k_tnll, k_ktail = extract_king_values(results)
        king_kl = king_kl or k_kl
        king_rkl = king_rkl or k_rkl
        king_eopd = king_eopd or k_eopd
        king_kl_is = king_kl_is or k_kl_is
        king_forking_rkl = king_forking_rkl or k_frk
        king_trace_nll = king_trace_nll or k_tnll
        king_kl_tail = king_kl_tail or k_ktail
    
    # Identify reference and teacher models
    reference_data = None
    teacher_data = None
    
    for model_name, student_data in students.items():
        if "reference" in model_name.lower():
            reference_data = student_data
        if "teacher" in model_name.lower():
            teacher_data = student_data
    
    # Resolve broken axes
    broken_axes = set()
    
    if reference_data:
        reference_broken = resolve_reference_broken_axes(reference_data)
        broken_axes.update(reference_broken)
        logger.info(f"Reference-broken axes: {reference_broken}")
    
    if teacher_data:
        teacher_broken = resolve_teacher_broken_axes(
            teacher_data, 
            king_kl=king_kl,
            king_rkl=king_rkl,
        )
        broken_axes.update(teacher_broken)
        logger.info(f"Teacher-broken axes: {teacher_broken}")
    
    # Compute teacher axes for reference comparison
    teacher_axes = None
    if teacher_data:
        teacher_axes = compute_axes(
            teacher_data,
            king_kl=king_kl,
            king_rkl=king_rkl,
            king_eopd=king_eopd,
            king_kl_is=king_kl_is,
            king_forking_rkl=king_forking_rkl,
            king_trace_nll=king_trace_nll,
            king_kl_tail=king_kl_tail,
            broken_axes=broken_axes,
        )
        logger.info(f"Teacher axes computed: {len(teacher_axes)} axes")
    
    # Compute reference axes for baseline penalty
    reference_axes = None
    if reference_data:
        reference_axes = compute_axes(
            reference_data,
            king_kl=king_kl,
            king_rkl=king_rkl,
            king_eopd=king_eopd,
            king_kl_is=king_kl_is,
            king_forking_rkl=king_forking_rkl,
            king_trace_nll=king_trace_nll,
            king_kl_tail=king_kl_tail,
            broken_axes=broken_axes,
        )
        logger.info(f"Reference axes computed: {len(reference_axes)} axes")
    
    # Compute composites for all students
    logger.info(f"Computing composite scores for {len(students)} models...")
    
    for model_name, student_data in students.items():
        try:
            composite = compute_composite(
                student_data,
                king_kl=king_kl,
                king_rkl=king_rkl,
                broken_axes=broken_axes if broken_axes else None,
                reference_axes=reference_axes,
                king_eopd=king_eopd,
                king_kl_is=king_kl_is,
                king_forking_rkl=king_forking_rkl,
                king_trace_nll=king_trace_nll,
                king_kl_tail=king_kl_tail,
                teacher_axes=teacher_axes,
            )
            student_data["composite"] = composite
            
            # Log composite scores
            logger.info(
                f"  {model_name}: final={composite.get('final')}, "
                f"worst={composite.get('worst')}, "
                f"weighted={composite.get('weighted')}"
            )
        except Exception as e:
            logger.error(f"Error computing composite for {model_name}: {e}", exc_info=True)
            student_data["composite"] = {
                "error": str(e),
                "version": None,
            }
    
    return results


def format_results_for_display(results: dict) -> dict:
    """Format results for clean display/export.
    
    Includes eval metrics and composite scores, organized by model.
    
    Args:
        results: Results dictionary with composite scores added
        
    Returns:
        Formatted results dictionary
    """
    formatted = {
        "metadata": results.get("metadata", {}),
        "models": {},
    }
    
    for model_name, student_data in results.get("students", {}).items():
        composite = student_data.get("composite", {})
        
        model_summary = {
            # Eval metrics
            "eval_metrics": {
                "kl_global_avg": student_data.get("kl_global_avg"),
                "capability": student_data.get("capability"),
                "length_axis": student_data.get("length_axis"),
                "think_probe": student_data.get("think_probe"),
                "on_policy_rkl": student_data.get("on_policy_rkl"),
                "judge_probe": student_data.get("judge_probe"),
                "chat_turns_probe": student_data.get("chat_turns_probe"),
            },
            # Per-axis composite scores
            "composite_axes": composite.get("axes"),
            # Aggregate composite scores
            "composite_scores": {
                "final": composite.get("final"),
                "worst": composite.get("worst"),
                "worst_3_mean": composite.get("worst_3_mean"),
                "weighted": composite.get("weighted"),
                "present_count": composite.get("present_count"),
            },
            # Composite metadata
            "composite_metadata": {
                "version": composite.get("version"),
                "axis_spread": composite.get("axis_spread"),
                "bench_vs_rel_gap": composite.get("bench_vs_rel_gap"),
                "broken_axes": composite.get("broken_axes"),
                "final_alpha": composite.get("final_alpha"),
            },
        }
        
        formatted["models"][model_name] = model_summary
    
    return formatted


def save_results(data: dict, output_path: str, pretty: bool = True) -> None:
    """Save results to JSON file.
    
    Args:
        data: Results dictionary to save
        output_path: Path to save results
        pretty: Whether to pretty-print JSON
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_path}")
    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=2 if pretty else None,
            default=str,  # Handle any non-serializable objects
        )
    logger.info(f"Results saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Compute composite scores from evaluation metrics"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON (from pod_eval_vllm.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results with composite scores (default: results_with_composite.json)",
    )
    parser.add_argument(
        "--output-formatted",
        type=str,
        default=None,
        help="Output path for formatted results (summary view)",
    )
    parser.add_argument(
        "--king-kl",
        type=float,
        default=None,
        help="King's KL value (auto-extracted if not provided)",
    )
    parser.add_argument(
        "--king-rkl",
        type=float,
        default=None,
        help="King's RKL value (auto-extracted if not provided)",
    )
    parser.add_argument(
        "--king-model",
        type=str,
        default=None,
        help="Model name to extract king metrics from (default: first model)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load results
        results = load_results(args.results)
        
        # Compute composite scores
        results_with_composites = compute_composites_for_all(
            results,
            king_kl=args.king_kl,
            king_rkl=args.king_rkl,
        )
        
        # Save full results
        output_path = args.output or "results_with_composite.json"
        save_results(results_with_composites, output_path)
        
        # Save formatted results if requested
        if args.output_formatted:
            formatted = format_results_for_display(results_with_composites)
            save_results(formatted, args.output_formatted)
            logger.info(f"Formatted results saved to {args.output_formatted}")
        
        # Print summary
        logger.info("\n" + "="*100)
        logger.info("COMPOSITE SCORE SUMMARY - RANKED BY FINAL SCORE")
        logger.info("="*100)
        
        models = results_with_composites.get("students", {})
        
        # Sort by final score (descending)
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1].get("composite", {}).get("final") or 0,
            reverse=True
        )
        
        for rank, (model_name, student_data) in enumerate(sorted_models, 1):
            composite = student_data.get("composite", {})
            
            if "error" in composite:
                logger.info(f"{rank:2d}. {model_name}: ERROR - {composite['error']}")
            else:
                final = composite.get("final")
                worst = composite.get("worst")
                worst_3_mean = composite.get("worst_3_mean")
                weighted = composite.get("weighted")
                present = composite.get("present_count", 0)
                axes = composite.get("axes", {})
                broken = composite.get("broken_axes", [])
                
                # Create visual bar for final score
                bar_width = int((final or 0) * 40)
                bar = "█" * bar_width + "░" * (40 - bar_width)
                
                # Print main scores
                logger.info(
                    f"\n{rank:2d}. {model_name:40s}"
                )
                logger.info(
                    f"    final={final:.4f} [{bar}] "
                    f"| worst_3_mean={worst_3_mean:.4f} | worst={worst:.4f} | weighted={weighted:.4f}"
                )
                logger.info(
                    f"    Axes: {present} present" +
                    (f" | Broken: {len(broken)}" if broken else "")
                )
                
                # Print sorted per-axis scores (only non-None axes)
                if axes:
                    sorted_axes = sorted(
                        [(k, v) for k, v in axes.items() if v is not None],
                        key=lambda x: x[1]
                    )
                    
                    logger.info(f"    Per-axis breakdown (sorted by score):")
                    for axis_name, axis_score in sorted_axes:
                        is_broken = " ⚠️ [broken]" if axis_name in broken else ""
                        axis_bar_width = int(axis_score * 25)
                        axis_bar = "█" * axis_bar_width + "░" * (25 - axis_bar_width)
                        logger.info(
                            f"      {axis_name:30s} {axis_score:.4f} [{axis_bar}]{is_broken}"
                        )
        
        # Print comparative ranking table
        logger.info("\n" + "="*100)
        logger.info("QUICK COMPARISON TABLE")
        logger.info("="*100)
        logger.info(
            f"{'Rank':<5} {'Model':<42} {'Final':<10} {'Worst-3':<10} {'Worst':<10} {'Weighted':<10} {'Axes':<6}"
        )
        logger.info("-" * 100)
        
        for rank, (model_name, student_data) in enumerate(sorted_models, 1):
            composite = student_data.get("composite", {})
            
            if "error" not in composite:
                final = composite.get("final") or 0
                worst_3 = composite.get("worst_3_mean") or 0
                worst = composite.get("worst") or 0
                weighted = composite.get("weighted") or 0
                present = composite.get("present_count", 0)
                
                logger.info(
                    f"{rank:<5} {model_name:<42} {final:<10.4f} {worst_3:<10.4f} "
                    f"{worst:<10.4f} {weighted:<10.4f} {present:<6}"
                )
        
        # Print per-axis comparison across all models
        logger.info("\n" + "="*100)
        logger.info("PER-AXIS COMPARISON - ALL MODELS")
        logger.info("="*100)
        
        # Collect all axes and their values for all models
        all_axes_data = {}
        
        for model_name, student_data in sorted_models:
            composite = student_data.get("composite", {})
            if "error" not in composite:
                axes = composite.get("axes", {})
                for axis_name, axis_score in axes.items():
                    if axis_score is not None:
                        if axis_name not in all_axes_data:
                            all_axes_data[axis_name] = []
                        all_axes_data[axis_name].append((model_name, axis_score))
        
        # Sort axes by average score (to show weak/strong axes first)
        axes_by_avg = sorted(
            all_axes_data.items(),
            key=lambda x: sum(score for _, score in x[1]) / len(x[1])
        )
        
        for axis_name, model_scores in axes_by_avg:
            # Sort models by score (descending) for this axis
            sorted_by_score = sorted(model_scores, key=lambda x: x[1], reverse=True)
            best_score = sorted_by_score[0][1]
            worst_score = sorted_by_score[-1][1]
            avg_score = sum(score for _, score in sorted_by_score) / len(sorted_by_score)
            gap = best_score - worst_score
            
            logger.info(f"\n{axis_name}:")
            logger.info(f"  Avg: {avg_score:.4f} | Range: {worst_score:.4f} - {best_score:.4f} | Gap: {gap:.4f}")
            logger.info(f"  " + "-" * 96)
            
            for rank, (model_name, score) in enumerate(sorted_by_score, 1):
                diff_from_best = score - best_score
                diff_pct = (diff_from_best / best_score * 100) if best_score > 0 else 0
                bar_width = int(score * 30)
                bar = "█" * bar_width + "░" * (30 - bar_width)
                
                # Show difference indicator
                if diff_from_best == 0:
                    diff_str = "       ✓ BEST"
                else:
                    diff_str = f"{diff_from_best:+.4f} ({diff_pct:+.1f}%)"
                
                logger.info(
                    f"    {rank}. {model_name:<42s} {score:.4f} [{bar}] {diff_str}"
                )
        
        # Print axis strength/weakness summary
        logger.info("\n" + "="*100)
        logger.info("AXIS STRENGTH SUMMARY (Sorted by Average Score)")
        logger.info("="*100)
        logger.info(
            f"{'Axis':<35} {'Avg':<10} {'Best':<10} {'Worst':<10} {'Gap':<10} {'Status':<15}"
        )
        logger.info("-" * 100)
        
        for axis_name, model_scores in axes_by_avg:
            scores = [score for _, score in model_scores]
            avg = sum(scores) / len(scores)
            best = max(scores)
            worst = min(scores)
            gap = best - worst
            
            # Classify axis strength
            if avg < 0.3:
                status = "🔴 WEAK"
            elif avg < 0.5:
                status = "🟠 FAIR"
            elif avg < 0.7:
                status = "🟡 GOOD"
            elif avg < 0.85:
                status = "🟢 STRONG"
            else:
                status = "🟢 EXCELLENT"
            
            logger.info(
                f"{axis_name:<35} {avg:<10.4f} {best:<10.4f} {worst:<10.4f} {gap:<10.4f} {status}"
            )
        
        logger.info("="*100)
        logger.info("✓ Composite score computation complete!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
