# evaluate_test.py
"""
Evaluate model on test set
"""
from pathlib import Path
import torch

from src.improved_model_v2 import RAF_DB_MultiAttribute_Net
from src.evaluate import ModelEvaluator, create_test_dataloader

def main():
    print("="*70)
    print("RAF-DB TEST SET EVALUATION")
    print("="*70)
    
    # Configuration
    checkpoint_path = 'checkpoints/rafdb_multi_attribute_final.pth'
    test_dir = 'data/test_organized'
    batch_size = 32
    device = 'cpu'
    
    # Check if test directory exists
    if not Path(test_dir).exists():
        print(f"\n❌ Test directory not found: {test_dir}")
        print("\nPlease organize your test data as:")
        print(f"  {test_dir}/")
        print("    ├── happy/")
        print("    ├── angry/")
        print("    ├── sad/")
        print("    ├── fear/")
        print("    ├── disgust/")
        print("    ├── surprise/")
        print("    └── neutral/")
        return
    
    # Load model
    print("\nLoading model...")
    model, checkpoint = RAF_DB_MultiAttribute_Net.load_from_checkpoint(
        checkpoint_path,
        device=device,
        model_name='resnet18'
    )
    
    # Create test dataloader
    print("\nLoading test data...")
    test_loader = create_test_dataloader(test_dir, batch_size=batch_size)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device=device)
    
    # Evaluate with correction
    print("\n" + "="*70)
    print("EVALUATION WITH MULTI-ATTRIBUTE CORRECTION")
    print("="*70)
    
    results_corrected = evaluator.evaluate(test_loader, use_correction=True)
    
    print(f"\n🎯 RESULTS (With Correction):")
    print(f"   Accuracy:  {results_corrected['accuracy']:.2f}%")
    print(f"   Precision: {results_corrected['precision']:.4f}")
    print(f"   Recall:    {results_corrected['recall']:.4f}")
    print(f"   F1-Score:  {results_corrected['f1_score']:.4f}")
    
    # Evaluate without correction (for comparison)
    print("\n" + "="*70)
    print("EVALUATION WITHOUT CORRECTION (Baseline)")
    print("="*70)
    
    results_baseline = evaluator.evaluate(test_loader, use_correction=False)
    
    print(f"\n🎯 RESULTS (Without Correction):")
    print(f"   Accuracy:  {results_baseline['accuracy']:.2f}%")
    print(f"   Precision: {results_baseline['precision']:.4f}")
    print(f"   Recall:    {results_baseline['recall']:.4f}")
    print(f"   F1-Score:  {results_baseline['f1_score']:.4f}")
    
    # Compare
    improvement = results_corrected['accuracy'] - results_baseline['accuracy']
    print(f"\n📈 IMPROVEMENT: {improvement:+.2f}%")
    
    if improvement > 0:
        print("   ✅ Multi-attribute correction improved accuracy!")
    else:
        print("   ℹ️  No significant improvement from correction")
    
    # Print classification report
    evaluator.print_classification_report(
        results_corrected['labels'],
        results_corrected['predictions']
    )
    
    # Analyze confusion patterns
    evaluator.analyze_confusion_patterns(
        results_corrected['labels'],
        results_corrected['predictions']
    )
    
    # Plot confusion matrix
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    evaluator.plot_confusion_matrix(
        results_corrected['labels'],
        results_corrected['predictions'],
        save_path='results/plots/confusion_matrix_test.png'
    )
    
    # Save results
    import json
    results_summary = {
        'with_correction': {
            'accuracy': float(results_corrected['accuracy']),
            'precision': float(results_corrected['precision']),
            'recall': float(results_corrected['recall']),
            'f1_score': float(results_corrected['f1_score'])
        },
        'without_correction': {
            'accuracy': float(results_baseline['accuracy']),
            'precision': float(results_baseline['precision']),
            'recall': float(results_baseline['recall']),
            'f1_score': float(results_baseline['f1_score'])
        },
        'improvement': float(improvement)
    }
    
    with open('results/evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n✅ Results saved:")
    print("   - results/plots/confusion_matrix_test.png")
    print("   - results/evaluation_summary.json")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()