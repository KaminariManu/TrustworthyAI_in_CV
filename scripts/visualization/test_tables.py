"""
Test script for generate_tables.py
Creates dummy data and tests all table generation functions.
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

from visualization.generate_tables import (
    load_results,
    create_summary_table,
    create_pivot_table,
    create_comparison_table,
    create_ranking_table,
    calculate_statistics,
    create_per_dataset_stats,
    create_per_model_stats,
    generate_all_tables
)


def create_dummy_results():
    """Create dummy experimental results for testing."""
    models = ['ResNet18', 'VGG16', 'MobileNetV2', 'DenseNet121']
    datasets = ['cifar10', 'gtsrb', 'mnist']
    
    results = []
    
    for model in models:
        for dataset in datasets:
            result = {
                'model': model,
                'dataset': dataset,
                'clean_accuracy': float(np.random.uniform(85, 95)),
                'attack_success_rate': float(np.random.uniform(70, 99)),
                'poison_ratio': 0.1,
                'num_poisoned': int(np.random.randint(1000, 5000)),
                'training_epochs': int(np.random.choice([20, 30, 50])),
                'timestamp': '2026-02-18_12:00:00'
            }
            results.append(result)
    
    return results


def test_table_functions(results):
    """Test individual table generation functions."""
    print("="*70)
    print("Test 1: Individual Table Functions")
    print("="*70)
    
    # Test summary table
    print("\n1. Testing create_summary_table...")
    try:
        summary = create_summary_table(results)
        print(f"   ✓ Summary table created with shape: {summary.shape}")
        print(summary.head(3).to_string(index=False))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test pivot table
    print("\n2. Testing create_pivot_table (Clean Accuracy)...")
    try:
        pivot_clean = create_pivot_table(results, metric='clean_accuracy')
        print(f"   ✓ Pivot table created with shape: {pivot_clean.shape}")
        print(pivot_clean.head(3).to_string())
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test comparison table
    print("\n3. Testing create_comparison_table...")
    try:
        comparison = create_comparison_table(results)
        print(f"   ✓ Comparison table created with shape: {comparison.shape}")
        print(comparison.head(2).to_string())
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test ranking table
    print("\n4. Testing create_ranking_table...")
    try:
        ranking = create_ranking_table(results, metric='attack_success_rate')
        print(f"   ✓ Ranking table created with shape: {ranking.shape}")
        print(ranking.head(5).to_string(index=False))
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test statistics
    print("\n5. Testing calculate_statistics...")
    try:
        stats = calculate_statistics(results)
        print(f"   ✓ Statistics calculated: {len(stats)} metrics")
        for key in list(stats.keys())[:5]:
            value = stats[key]
            if isinstance(value, float):
                print(f"      {key}: {value:.2f}")
            else:
                print(f"      {key}: {value}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test per-dataset stats
    print("\n6. Testing create_per_dataset_stats...")
    try:
        dataset_stats = create_per_dataset_stats(results)
        print(f"   ✓ Per-dataset stats created with shape: {dataset_stats.shape}")
        print(dataset_stats.to_string())
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test per-model stats
    print("\n7. Testing create_per_model_stats...")
    try:
        model_stats = create_per_model_stats(results)
        print(f"   ✓ Per-model stats created with shape: {model_stats.shape}")
        print(model_stats.head(2).to_string())
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n✓ All individual function tests passed!")
    return True


def test_generate_all_tables(results_path):
    """Test the main generate_all_tables function."""
    print("\n" + "="*70)
    print("Test 2: Generate All Tables")
    print("="*70)
    
    try:
        tables = generate_all_tables(
            results_path=results_path,
            save_dir='./test_results/tables',
            attack_name='Test Attack'
        )
        
        print("\n✓ All tables generated successfully!")
        print(f"✓ Generated {len(tables)} table types")
        
        return True
    except Exception as e:
        print(f"\n✗ Error generating tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(results, test_dir='./test_results/tables'):
    """Test saving and loading results."""
    print("\n" + "="*70)
    print("Test 3: Save and Load Results")
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
    print("✓ Results match!")
    
    return results_path


def verify_output_files(save_dir='./test_results/tables'):
    """Verify that all expected files were created."""
    print("\n" + "="*70)
    print("Test 4: Verify Output Files")
    print("="*70)
    
    expected_extensions = ['.csv', '.tex', '.md']
    found_files = {'csv': 0, 'tex': 0, 'md': 0}
    
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            for ext in expected_extensions:
                if file.endswith(ext):
                    found_files[ext[1:]] += 1
    
    print(f"\n✓ Found {found_files['csv']} CSV files")
    print(f"✓ Found {found_files['tex']} LaTeX files")
    print(f"✓ Found {found_files['md']} Markdown files")
    
    total = sum(found_files.values())
    print(f"\n✓ Total: {total} output files created")
    
    return total > 0


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
    print("TABLE GENERATION TESTING SUITE")
    print("="*70)
    
    # Create dummy data
    print("\nCreating dummy experimental results...")
    results = create_dummy_results()
    print(f"✓ Created {len(results)} dummy results")
    print(f"  - Models: {len(set(r['model'] for r in results))} unique")
    print(f"  - Datasets: {len(set(r['dataset'] for r in results))} unique")
    
    # Test 1: Individual functions
    try:
        success = test_table_functions(results)
        if not success:
            print("\n✗ Some function tests failed")
            return
    except Exception as e:
        print(f"\n✗ Function tests failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Save/Load
    try:
        results_path = test_save_load(results)
    except Exception as e:
        print(f"\n✗ Save/Load failed: {e}")
        return
    
    # Test 3: Generate all tables
    try:
        success = test_generate_all_tables(results_path)
        if not success:
            print("\n✗ Table generation failed")
            return
    except Exception as e:
        print(f"\n✗ Table generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Verify outputs
    try:
        success = verify_output_files()
        if not success:
            print("\n✗ No output files found")
            return
    except Exception as e:
        print(f"\n✗ File verification failed: {e}")
        return
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ All tests passed successfully!")
    print(f"✓ Test results saved to: ./test_results/tables/")
    print(f"✓ Check subdirectories: csv files, latex/, markdown/")
    print("="*70)
    
    # Cleanup
    cleanup_test_files()


if __name__ == "__main__":
    main()
