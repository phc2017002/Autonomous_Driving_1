import os
import argparse
import json
import re
from tqdm import tqdm
from datetime import datetime

##########################################################################

def parse_grounded_caption_file(file_path):
    """
    Parses a grounded caption text file to extract phrases and the cleaned full text.
    Handles tags like <car>...</car> and <accident_object_1>...</accident_object_1>.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find all tags and their content, e.g., <tag>content</tag>
    tag_regex = re.compile(r"<(\w+[^>]*)>(.*?)</\1>", re.DOTALL)

    # Extract all phrases from within the tags
    matches = tag_regex.findall(content)
    phrases = [match[1].strip() for match in matches]

    # Create a cleaned version of the text by removing tags but keeping content
    cleaned_text = tag_regex.sub(r'\2', content)

    # Clean up headers and extra whitespace
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.match(r'^(VIDEO INFO:|SECTION|ACCIDENT|PRE-ACCIDENT|---)', line.strip()):
            cleaned_lines.append(line.strip())

    # Re-join the text and clean up multiple spaces/newlines
    cleaned_text = ' '.join(cleaned_lines).strip()
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('At **', 'At')

    return cleaned_text, phrases

##########################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GCG Task - Caption Only")
    
    # Arguments for text file locations
    parser.add_argument("--gt_caption_dir", required=True, type=str, 
                        help="Base directory for ground truth caption .txt files.")
    parser.add_argument("--pred_caption_dir", required=True, type=str, 
                        help="Base directory for prediction caption .txt files.")

    # Evaluation flags for caption quality
    parser.add_argument("--eval_caption", action="store_true", default=False, 
                        help="Evaluate caption quality using COCO metrics.")
    parser.add_argument("--use_clair", action="store_true", default=False, 
                        help="Evaluate caption quality using CLAIR.")
    parser.add_argument("--skip_spice", action="store_true", default=False,
                        help="Skip SPICE evaluation to avoid memory issues.")
    parser.add_argument("--max_caption_length", type=int, default=500,
                        help="Maximum caption length in words (to avoid memory issues). Default: 500")
    
    # Output file argument
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results (default: evaluation_results.json)")

    return parser.parse_args()

##########################################################################

def compute_metrics_for_pair(gt_caption, pred_caption, skip_spice=False, max_length=500):
    """Compute metrics for a single GT-prediction pair"""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        if not skip_spice:
            from pycocoevalcap.spice.spice import Spice
    except ImportError:
        print("Please install pycocotools and pycocoevalcap to evaluate caption quality:")
        print("pip install pycocotools pycocoevalcap")
        return None
    
    # Truncate if needed
    gt_words = gt_caption.split()
    pred_words = pred_caption.split()
    
    if len(gt_words) > max_length:
        gt_caption = ' '.join(gt_words[:max_length])
    if len(pred_words) > max_length:
        pred_caption = ' '.join(pred_words[:max_length])
    
    # Format for scorers
    gts = {0: [gt_caption]}
    res = {0: [pred_caption]}
    
    scores = {}
    
    # Compute BLEU
    scorer = Bleu(4)
    score, _ = scorer.compute_score(gts, res)
    for i, s in enumerate(score):
        scores[f'Bleu_{i+1}'] = s
    
    # Compute METEOR
    scorer = Meteor()
    score, _ = scorer.compute_score(gts, res)
    scores['METEOR'] = score
    
    # Compute ROUGE
    scorer = Rouge()
    score, _ = scorer.compute_score(gts, res)
    scores['ROUGE_L'] = score
    
    # Compute CIDEr
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    scores['CIDEr'] = score
    
    # Compute SPICE if not skipped
    if not skip_spice:
        try:
            scorer = Spice()
            score, _ = scorer.compute_score(gts, res)
            scores['SPICE'] = score
        except Exception as e:
            print(f"SPICE computation failed: {e}")
            scores['SPICE'] = None
    
    return scores

##########################################################################

def eval_caption_quality(all_gt_references, all_pred_captions, video_ids, skip_spice=False, max_length=500):
    """Evaluate using COCO metrics and return scores per video and aggregate"""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        if not skip_spice:
            from pycocoevalcap.spice.spice import Spice
    except ImportError:
        print("Please install pycocotools and pycocoevalcap to evaluate caption quality:")
        print("pip install pycocotools pycocoevalcap")
        return None

    per_video_scores = {}
    valid_videos = []
    
    print("Computing per-video scores...")
    for video_id, gt_ref, pred_caption in tqdm(zip(video_ids, all_gt_references, all_pred_captions), 
                                                total=len(video_ids), desc="Evaluating videos"):
        if not gt_ref or not pred_caption:
            per_video_scores[video_id] = {
                'status': 'skipped',
                'reason': 'empty_caption',
                'scores': {}
            }
            continue
        
        try:
            scores = compute_metrics_for_pair(gt_ref, pred_caption, skip_spice, max_length)
            per_video_scores[video_id] = {
                'status': 'success',
                'scores': scores,
                'gt_length': len(gt_ref.split()),
                'pred_length': len(pred_caption.split())
            }
            valid_videos.append(video_id)
        except Exception as e:
            per_video_scores[video_id] = {
                'status': 'error',
                'error': str(e),
                'scores': {}
            }
    
    # Compute aggregate scores across all valid videos
    print("\nComputing aggregate scores across all videos...")
    
    if not valid_videos:
        print("No valid videos to compute aggregate scores.")
        return {
            'per_video_scores': per_video_scores,
            'aggregate_scores': {},
            'spice_skipped': skip_spice
        }
    
    # Prepare data for aggregate scoring
    gts = {}
    res = {}
    
    for idx, video_id in enumerate(valid_videos):
        vid_idx = video_ids.index(video_id)
        gt_ref = all_gt_references[vid_idx]
        pred_caption = all_pred_captions[vid_idx]
        
        # Truncate to max_length
        gt_words = gt_ref.split()
        pred_words = pred_caption.split()
        
        if len(gt_words) > max_length:
            gt_ref = ' '.join(gt_words[:max_length])
        if len(pred_words) > max_length:
            pred_caption = ' '.join(pred_words[:max_length])
        
        gts[idx] = [gt_ref]
        res[idx] = [pred_caption]
    
    # Compute aggregate scores
    aggregate_scores = {}
    
    # Set up scorers
    scorers = {
        'Bleu': Bleu(4),
        'METEOR': Meteor(),
        'ROUGE_L': Rouge(),
        'CIDEr': Cider()
    }
    
    if not skip_spice:
        scorers['SPICE'] = Spice()
    
    for scorer_name, scorer in scorers.items():
        print(f'Computing aggregate {scorer_name} score...')
        try:
            score, _ = scorer.compute_score(gts, res)
            
            if scorer_name == 'Bleu':
                for i, s in enumerate(score):
                    aggregate_scores[f'Bleu_{i+1}'] = s
            else:
                aggregate_scores[scorer_name] = score
        except Exception as e:
            print(f"Error computing {scorer_name}: {e}")
            if scorer_name == 'Bleu':
                for i in range(4):
                    aggregate_scores[f'Bleu_{i+1}'] = None
            else:
                aggregate_scores[scorer_name] = None
    
    # Calculate average of individual scores as well
    avg_scores = {}
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
    if not skip_spice:
        metrics.append('SPICE')
    
    for metric in metrics:
        scores = [per_video_scores[vid]['scores'].get(metric, 0) 
                 for vid in valid_videos 
                 if per_video_scores[vid]['scores'].get(metric) is not None]
        if scores:
            avg_scores[f'{metric}_avg'] = sum(scores) / len(scores)
    
    # Print results
    print("\n" + "="*50)
    print("Aggregate COCO Captioning Metrics:")
    print("="*50)
    for metric, score in aggregate_scores.items():
        if score is not None:
            print(f'\033[92m{metric:<15}: {score:.4f}\033[0m')
    print("\n" + "="*50)
    print("Average of Individual Scores:")
    print("="*50)
    for metric, score in avg_scores.items():
        print(f'\033[92m{metric:<15}: {score:.4f}\033[0m')
    print("="*50)
    
    return {
        'per_video_scores': per_video_scores,
        'aggregate_scores': aggregate_scores,
        'average_individual_scores': avg_scores,
        'spice_skipped': skip_spice,
        'valid_videos': valid_videos
    }

##########################################################################
# MAIN EXECUTION
##########################################################################

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the results dictionary
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'gt_caption_dir': args.gt_caption_dir,
            'pred_caption_dir': args.pred_caption_dir,
            'max_caption_length': args.max_caption_length,
            'skip_spice': args.skip_spice
        },
        'video_processing': {},
        'metrics': {}
    }
    
    # Initialize lists to store caption text for all videos
    all_gt_references = []
    all_pred_captions = []
    video_processing_details = []
    processed_video_ids = []
    
    # Get list of video IDs from the ground truth directory
    video_ids = sorted([d for d in os.listdir(args.gt_caption_dir) 
                       if os.path.isdir(os.path.join(args.gt_caption_dir, d))])
    
    if not video_ids:
        print(f"No video directories found in {args.gt_caption_dir}")
        exit(1)
    
    print(f"Found {len(video_ids)} videos to evaluate: {video_ids[:5]}...")
    all_results['video_ids'] = video_ids
    
    # Process each video
    for video_id in tqdm(video_ids, desc="Processing videos"):
        video_detail = {'video_id': video_id, 'status': 'success'}
        
        try:
            # Define paths for GT and Prediction caption files
            gt_caption_file = os.path.join(args.gt_caption_dir, video_id, 
                                          'grounded_captions', 
                                          f'{video_id}_grounded_caption.txt')
            pred_caption_file = os.path.join(args.pred_caption_dir, video_id, 
                                            'grounded_captions', 
                                            f'{video_id}_grounded_caption.txt')
            
            # Check if both files exist
            if not os.path.exists(gt_caption_file):
                print(f"Warning: GT file not found for {video_id}: {gt_caption_file}")
                video_detail['status'] = 'gt_not_found'
                video_detail['error'] = f"GT file not found: {gt_caption_file}"
                video_processing_details.append(video_detail)
                all_gt_references.append("")
                all_pred_captions.append("")
                processed_video_ids.append(video_id)
                continue
                
            if not os.path.exists(pred_caption_file):
                print(f"Warning: Pred file not found for {video_id}: {pred_caption_file}")
                # Add GT but empty prediction for this case
                gt_text_cleaned, _ = parse_grounded_caption_file(gt_caption_file)
                all_gt_references.append(gt_text_cleaned)
                all_pred_captions.append("")
                video_detail['status'] = 'pred_not_found'
                video_detail['error'] = f"Pred file not found: {pred_caption_file}"
                video_processing_details.append(video_detail)
                processed_video_ids.append(video_id)
                continue
            
            # Parse the text files to get the cleaned text for evaluation
            gt_text_cleaned, _ = parse_grounded_caption_file(gt_caption_file)
            pred_text_cleaned, _ = parse_grounded_caption_file(pred_caption_file)
            
            all_gt_references.append(gt_text_cleaned)
            all_pred_captions.append(pred_text_cleaned)
            
            video_detail['gt_length'] = len(gt_text_cleaned.split())
            video_detail['pred_length'] = len(pred_text_cleaned.split())
            video_processing_details.append(video_detail)
            processed_video_ids.append(video_id)
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            video_detail['status'] = 'error'
            video_detail['error'] = str(e)
            video_processing_details.append(video_detail)
            all_gt_references.append("")
            all_pred_captions.append("")
            processed_video_ids.append(video_id)
            continue
    
    all_results['video_processing']['details'] = video_processing_details
    all_results['video_processing']['total_videos'] = len(video_ids)
    all_results['video_processing']['successful_pairs'] = len([d for d in video_processing_details if d['status'] == 'success'])
    
    print(f"\nSuccessfully loaded {all_results['video_processing']['successful_pairs']} caption pairs for evaluation.")
    
    # Evaluate Caption Quality with COCO metrics
    if args.eval_caption:
        if not any(pred for pred in all_pred_captions if pred):
            print("No valid predictions to evaluate for caption quality.")
            all_results['metrics']['coco'] = {'error': 'No valid caption pairs'}
        else:
            coco_results = eval_caption_quality(all_gt_references, all_pred_captions, processed_video_ids,
                                               skip_spice=args.skip_spice, 
                                               max_length=args.max_caption_length)
            if coco_results:
                all_results['metrics']['coco'] = coco_results
    
    # If no evaluation method was specified, show a warning
    if not args.eval_caption and not args.use_clair:
        print("\nWarning: No evaluation method specified!")
        print("Use --eval_caption for COCO metrics or --use_clair for CLAIR evaluation.")
        all_results['warning'] = 'No evaluation method specified'
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results to file
    output_path = args.output_file
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n" + "="*50)
    print(f"Results saved to: {output_path}")
    print("="*50)
    
    # Also create a summary text file for quick viewing
    summary_path = output_path.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Timestamp: {all_results['timestamp']}\n")
        f.write(f"GT Directory: {args.gt_caption_dir}\n")
        f.write(f"Pred Directory: {args.pred_caption_dir}\n")
        f.write(f"Total Videos: {all_results['video_processing']['total_videos']}\n")
        f.write(f"Successful Pairs: {all_results['video_processing']['successful_pairs']}\n\n")
        
        if 'coco' in all_results['metrics']:
            if 'aggregate_scores' in all_results['metrics']['coco']:
                f.write("Aggregate COCO Metrics:\n")
                f.write("-"*30 + "\n")
                for metric, score in all_results['metrics']['coco']['aggregate_scores'].items():
                    if score is not None:
                        f.write(f"{metric:<15}: {score:.4f}\n")
                f.write("\n")
            
            if 'per_video_scores' in all_results['metrics']['coco']:
                f.write("Per-Video COCO Metrics:\n")
                f.write("-"*30 + "\n")
                for video_id, data in all_results['metrics']['coco']['per_video_scores'].items():
                    if data['status'] == 'success' and data['scores']:
                        f.write(f"\n{video_id}:\n")
                        for metric, score in data['scores'].items():
                            if score is not None:
                                f.write(f"  {metric:<12}: {score:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")