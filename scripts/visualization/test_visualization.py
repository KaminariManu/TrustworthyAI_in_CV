"""
Test script for visualization utilities and generate_figures.
Creates dummy data and tests visualization functions.
"""

import os
import sys
import json
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add scripts directory to path
scripts_dir = os.path.join(project_root, 'scripts')
sys.path.insert(0, scripts_dir)

from visualization.generate_figures import (
    load_results,
    plot_accuracy_comparison,
    plot_asr_comparison,
    plot_heatmap_clean_accuracy,
    plot_heatmap_asr,
    plot_scatter_clean_vs_asr,
    plot_model_comparison_grouped
)


def create_dummy_results():
    """Create dummy experimental results for testing."""
    models = ['ResNet18', 'VGG16', 'MobileNetV2']
    datasets = ['cifar10', 'gtsrb']
    
    results = []
    
    for model in models:
        for dataset in datasets:
            result = {
                'model': model,
                'dataset': dataset,
                'clean_accuracy': np.random.uniform(85, 95),
                'attack_success_rate': np.random.uniform(70, 99),
                'poisoning_rate': 0.1,
                'timestamp': '2026-02-18_12:00:00'
            }
            results.append(result)
    
    return results


def test_save_load_results(results, test_dir='./test_results'):
    """Test saving and loading results."""
    print("="*70)
    print("Test 1: Save and Load Results")
    print("="*70)
    
    os.makedirs(test_dir, exist_ok=True)
    
    # Save results to JSON
    results_path = os.path.join(test_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {results_path}")
    
    # Load results
    loaded_results = load_results(results_path)
    print(f"✓ Loaded {len(loaded_results)} results")
    
    assert len(loaded_results) == len(results), "Mismatch in number of results"
    print("✓ Results match!\n")
    
    return results_path


def test_visualization_functions(results, save_dir='./test_results/figures'):
    """Test all visualization functions."""
    print("="*70)
    print("Test 2: Visualization Functions")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    attack_name = "Test Attack"
    
    # Test 1: Clean Accuracy Comparison
    print("\n1. Testing plot_accuracy_comparison...")
    try:
        plot_accuracy_comparison(
            results,
            os.path.join(save_dir, 'test_clean_accuracy_comparison.png'),
            attack_name
        )
        print("   ✓ Clean accuracy comparison plot created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: ASR Comparison
    print("2. Testing plot_asr_comparison...")
    try:
        plot_asr_comparison(
            results,
            os.path.join(save_dir, 'test_asr_comparison.png'),
            attack_name
        )
        print("   ✓ ASR comparison plot created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3: Clean Accuracy Heatmap
    print("3. Testing plot_heatmap_clean_accuracy...")
    try:
        plot_heatmap_clean_accuracy(
            results,
            os.path.join(save_dir, 'test_clean_accuracy_heatmap.png'),
            attack_name
        )
        print("   ✓ Clean accuracy heatmap created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 4: ASR Heatmap
    print("4. Testing plot_heatmap_asr...")
    try:
        plot_heatmap_asr(
            results,
            os.path.join(save_dir, 'test_asr_heatmap.png'),
            attack_name
        )
        print("   ✓ ASR heatmap created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 5: Scatter Plot
    print("5. Testing plot_scatter_clean_vs_asr...")
    try:
        plot_scatter_clean_vs_asr(
            results,
            os.path.join(save_dir, 'test_scatter_clean_vs_asr.png'),
            attack_name
        )
        print("   ✓ Scatter plot created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 6: Model Comparison
    print("6. Testing plot_model_comparison_grouped...")
    try:
        plot_model_comparison_grouped(
            results,
            os.path.join(save_dir, 'test_model_comparison.png'),
            attack_name
        )
        print("   ✓ Model comparison plot created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n✓ All visualization tests passed!")
    return True


def test_visualization_utils():
    """Test visualization utilities independently."""
    print("\n" + "="*70)
    print("Test 3: Visualization Utilities")
    print("="*70)
    
    from utils.visualization_utils import (
        plot_metric_heatmap,
        plot_scatter_with_annotations,
        save_results_table,
        create_results_summary
    )
    import pandas as pd
    
    # Create test data
    test_data = {
        'model': ['ResNet18', 'VGG16', 'MobileNetV2'] * 2,
        'dataset': ['cifar10'] * 3 + ['gtsrb'] * 3,
        'clean_accuracy': [90.5, 88.2, 89.7, 87.3, 85.1, 86.9],
        'attack_success_rate': [95.2, 93.8, 94.5, 96.1, 94.3, 95.7]
    }
    df = pd.DataFrame(test_data)
    
    save_dir = './test_results/utils_figures'
    os.makedirs(save_dir, exist_ok=True)
    
    # Test heatmap
    print("\n1. Testing plot_metric_heatmap...")
    try:
        plot_metric_heatmap(
            df,
            metric='clean_accuracy',
            row_var='model',
            col_var='dataset',
            save_path=os.path.join(save_dir, 'test_util_heatmap.png'),
            show=False
        )
        print("   ✓ Heatmap created using utils")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test scatter
    print("2. Testing plot_scatter_with_annotations...")
    try:
        plot_scatter_with_annotations(
            df,
            x_metric='clean_accuracy',
            y_metric='attack_success_rate',
            label_col='model',
            hue_col='dataset',
            save_path=os.path.join(save_dir, 'test_util_scatter.png'),
            show=False
        )
        print("   ✓ Scatter plot created using utils")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test results table
    print("3. Testing save_results_table...")
    try:
        save_results_table(
            df,
            os.path.join('./test_results', 'test_results_table.csv')
        )
        print("   ✓ Results table saved")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n✓ All utility tests passed!")
    return True


def cleanup_test_files():
    """Optionally cleanup test files."""
    import shutil
    
    print("\n" + "="*70)
    print("Cleanup")
    print("="*70)
    
    response = input("\nDo you want to delete test files? (y/n): ").strip().lower()
    
    if response == 'y':
        if os.path.exists('./test_results'):
            shutil.rmtree('./test_results')
            print("✓ Test files deleted")
    else:
        print("✓ Test files kept in ./test_results/")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VISUALIZATION TESTING SUITE")
    print("="*70)
    
    # Create dummy data
    print("\nCreating dummy experimental results...")
    results = create_dummy_results()
    print(f"✓ Created {len(results)} dummy results")
    
    # Test 1: Save/Load
    try:
        results_path = test_save_load_results(results)
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # Test 2: Visualization functions
    try:
        success = test_visualization_functions(results)
        if not success:
            print("\n✗ Some visualization tests failed")
            return
    except Exception as e:
        print(f"\n✗ Visualization tests failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Utilities
    try:
        success = test_visualization_utils()
        if not success:
            print("\n✗ Some utility tests failed")
            return
    except Exception as e:
        print(f"\n✗ Utility tests failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ All tests passed successfully!")
    print(f"✓ Test results saved to: ./test_results/")
    print(f"✓ Test figures saved to: ./test_results/figures/")
    print("="*70)
    
    # Cleanup
    cleanup_test_files()


if __name__ == "__main__":
    main()
