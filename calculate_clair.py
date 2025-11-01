import os
import argparse
import json
import re
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

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

class LocalCLAIR:
    """
    Local implementation of CLAIR-like metric without API dependencies.
    Uses sentence transformers and BERT models for semantic similarity.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        """
        Initialize the local CLAIR evaluator.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on (cuda/cpu)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing LocalCLAIR with model: {model_name} on device: {self.device}")
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer(model_name, device=self.device)
        
        # Load BERT for additional contextual understanding
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()
    
    def get_sentence_embedding(self, text):
        """Get sentence embedding using sentence transformer."""
        return self.sentence_model.encode(text, convert_to_tensor=False)
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for a text."""
        inputs = self.bert_tokenizer(text, return_tensors="pt", 
                                     max_length=512, truncation=True, 
                                     padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def compute_semantic_similarity(self, pred_text, gt_text):
        """Compute semantic similarity between prediction and ground truth."""
        # Get embeddings
        pred_emb = self.get_sentence_embedding(pred_text)
        gt_emb = self.get_sentence_embedding(gt_text)
        
        # Compute cosine similarity
        similarity = cosine_similarity([pred_emb], [gt_emb])[0, 0]
        
        return similarity
    
    def compute_contextual_similarity(self, pred_text, gt_text):
        """Compute contextual similarity using BERT."""
        pred_emb = self.get_bert_embedding(pred_text)
        gt_emb = self.get_bert_embedding(gt_text)
        
        similarity = cosine_similarity([pred_emb], [gt_emb])[0, 0]
        
        return similarity
    
    def compute_ngram_overlap(self, pred_text, gt_text, n=2):
        """Compute n-gram overlap between texts."""
        def get_ngrams(text, n):
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        pred_ngrams = get_ngrams(pred_text, n)
        gt_ngrams = get_ngrams(gt_text, n)
        
        if not gt_ngrams:
            return 0.0
        
        overlap = len(pred_ngrams.intersection(gt_ngrams))
        return overlap / len(gt_ngrams)
    
    def compute_information_coverage(self, pred_text, gt_text):
        """
        Compute how much information from GT is covered in prediction.
        This is a simplified version of information extraction.
        """
        # Split into sentences
        gt_sentences = [s.strip() for s in gt_text.split('.') if s.strip()]
        pred_sentences = [s.strip() for s in pred_text.split('.') if s.strip()]
        
        if not gt_sentences:
            return 0.0
        
        # For each GT sentence, find the best matching pred sentence
        coverage_scores = []
        for gt_sent in gt_sentences:
            if not gt_sent:
                continue
            
            best_score = 0.0
            for pred_sent in pred_sentences:
                if not pred_sent:
                    continue
                
                # Compute similarity between sentences
                score = self.compute_semantic_similarity(pred_sent, gt_sent)
                best_score = max(best_score, score)
            
            coverage_scores.append(best_score)
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def compute_clair_score(self, pred_text, gt_text):
        """
        Compute the final CLAIR-like score combining multiple metrics.
        
        Returns:
            score: Float between 0 and 1
            components: Dictionary with individual component scores
        """
        # Compute individual components
        semantic_sim = self.compute_semantic_similarity(pred_text, gt_text)
        contextual_sim = self.compute_contextual_similarity(pred_text, gt_text)
        bigram_overlap = self.compute_ngram_overlap(pred_text, gt_text, n=2)
        trigram_overlap = self.compute_ngram_overlap(pred_text, gt_text, n=3)
        info_coverage = self.compute_information_coverage(pred_text, gt_text)
        
        # Weighted combination (you can adjust these weights)
        weights = {
            'semantic': 0.3,
            'contextual': 0.2,
            'bigram': 0.15,
            'trigram': 0.15,
            'coverage': 0.2
        }
        
        final_score = (
            weights['semantic'] * semantic_sim +
            weights['contextual'] * contextual_sim +
            weights['bigram'] * bigram_overlap +
            weights['trigram'] * trigram_overlap +
            weights['coverage'] * info_coverage
        )
        
        components = {
            'semantic_similarity': float(semantic_sim),
            'contextual_similarity': float(contextual_sim),
            'bigram_overlap': float(bigram_overlap),
            'trigram_overlap': float(trigram_overlap),
            'information_coverage': float(info_coverage),
            'final_score': float(final_score)
        }
        
        return final_score, components

##########################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Captions with Local CLAIR Metric")
    
    # Arguments for text file locations
    parser.add_argument("--gt_caption_dir", required=True, type=str, 
                        help="Base directory for ground truth caption .txt files.")
    parser.add_argument("--pred_caption_dir", required=True, type=str, 
                        help="Base directory for prediction caption .txt files.")
    
    # Model selection
    parser.add_argument("--sentence_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model to use (default: all-MiniLM-L6-v2)")
    
    # Processing options
    parser.add_argument("--max_caption_length", type=int, default=500,
                        help="Maximum caption length in words. Default: 500")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Default: auto-detect")
    
    # Output file argument
    parser.add_argument("--output_file", type=str, default="clair_evaluation_results.json",
                        help="Path to save evaluation results (default: clair_evaluation_results.json)")

    return parser.parse_args()

##########################################################################

def evaluate_with_local_clair(gt_captions, pred_captions, video_ids, clair_evaluator, max_length=500):
    """
    Evaluate captions using the local CLAIR implementation.
    
    Returns:
        Dictionary containing per-video scores and aggregate statistics
    """
    per_video_results = {}
    all_scores = []
    all_components = {
        'semantic_similarity': [],
        'contextual_similarity': [],
        'bigram_overlap': [],
        'trigram_overlap': [],
        'information_coverage': []
    }
    
    print("\nEvaluating with Local CLAIR...")
    print("-" * 50)
    
    for video_id, gt_text, pred_text in tqdm(zip(video_ids, gt_captions, pred_captions),
                                             total=len(video_ids), desc="Computing CLAIR scores"):
        
        if not gt_text or not pred_text:
            per_video_results[video_id] = {
                'status': 'skipped',
                'reason': 'empty_caption',
                'score': None
            }
            continue
        
        try:
            # Truncate if needed
            gt_words = gt_text.split()
            pred_words = pred_text.split()
            
            if len(gt_words) > max_length:
                gt_text = ' '.join(gt_words[:max_length])
            if len(pred_words) > max_length:
                pred_text = ' '.join(pred_words[:max_length])
            
            # Compute CLAIR score
            score, components = clair_evaluator.compute_clair_score(pred_text, gt_text)
            
            per_video_results[video_id] = {
                'status': 'success',
                'score': float(score),
                'components': components,
                'gt_length': len(gt_words),
                'pred_length': len(pred_words)
            }
            
            all_scores.append(score)
            for key in all_components:
                if key in components:
                    all_components[key].append(components[key])
            
        except Exception as e:
            print(f"Error evaluating {video_id}: {e}")
            per_video_results[video_id] = {
                'status': 'error',
                'error': str(e),
                'score': None
            }
    
    # Compute aggregate statistics
    aggregate_stats = {}
    if all_scores:
        aggregate_stats['mean_score'] = float(np.mean(all_scores))
        aggregate_stats['std_score'] = float(np.std(all_scores))
        aggregate_stats['min_score'] = float(np.min(all_scores))
        aggregate_stats['max_score'] = float(np.max(all_scores))
        aggregate_stats['median_score'] = float(np.median(all_scores))
        
        # Component-wise statistics
        aggregate_stats['component_means'] = {}
        aggregate_stats['component_stds'] = {}
        for key, values in all_components.items():
            if values:
                aggregate_stats['component_means'][key] = float(np.mean(values))
                aggregate_stats['component_stds'][key] = float(np.std(values))
    
    # Print summary
    print("\n" + "="*60)
    print("LOCAL CLAIR EVALUATION SUMMARY:")
    print("="*60)
    
    if all_scores:
        print(f"Evaluated {len(all_scores)} video pairs successfully")
        print(f"\nOverall Statistics:")
        print(f"  Mean Score:   {aggregate_stats['mean_score']:.4f} ± {aggregate_stats['std_score']:.4f}")
        print(f"  Median Score: {aggregate_stats['median_score']:.4f}")
        print(f"  Min Score:    {aggregate_stats['min_score']:.4f}")
        print(f"  Max Score:    {aggregate_stats['max_score']:.4f}")
        
        print(f"\nComponent-wise Mean Scores:")
        for key, value in aggregate_stats['component_means'].items():
            if key != 'final_score':
                print(f"  {key:<25}: {value:.4f}")
    else:
        print("No valid video pairs to evaluate")
    
    print("="*60)
    
    return {
        'per_video_results': per_video_results,
        'aggregate_statistics': aggregate_stats,
        'num_evaluated': len(all_scores),
        'num_total': len(video_ids)
    }

##########################################################################
# MAIN EXECUTION
##########################################################################

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'gt_caption_dir': args.gt_caption_dir,
            'pred_caption_dir': args.pred_caption_dir,
            'sentence_model': args.sentence_model,
            'max_caption_length': args.max_caption_length,
            'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        }
    }
    
    # Initialize CLAIR evaluator
    print("Initializing Local CLAIR evaluator...")
    clair_evaluator = LocalCLAIR(model_name=args.sentence_model, device=args.device)
    
    # Get list of video IDs
    video_ids = sorted([d for d in os.listdir(args.gt_caption_dir) 
                       if os.path.isdir(os.path.join(args.gt_caption_dir, d))])
    
    if not video_ids:
        print(f"No video directories found in {args.gt_caption_dir}")
        exit(1)
    
    print(f"\nFound {len(video_ids)} videos to evaluate")
    results['video_ids'] = video_ids
    
    # Load all captions
    all_gt_captions = []
    all_pred_captions = []
    processing_status = []
    
    print("\nLoading captions...")
    for video_id in tqdm(video_ids, desc="Loading"):
        status = {'video_id': video_id}
        
        gt_caption_file = os.path.join(args.gt_caption_dir, video_id, 
                                      'grounded_captions', 
                                      f'{video_id}_grounded_caption.txt')
        pred_caption_file = os.path.join(args.pred_caption_dir, video_id, 
                                        'grounded_captions', 
                                        f'{video_id}_grounded_caption.txt')
        
        # Load GT
        if os.path.exists(gt_caption_file):
            gt_text, _ = parse_grounded_caption_file(gt_caption_file)
            status['gt_loaded'] = True
        else:
            gt_text = ""
            status['gt_loaded'] = False
            print(f"Warning: GT not found for {video_id}")
        
        # Load Prediction
        if os.path.exists(pred_caption_file):
            pred_text, _ = parse_grounded_caption_file(pred_caption_file)
            status['pred_loaded'] = True
        else:
            pred_text = ""
            status['pred_loaded'] = False
            print(f"Warning: Prediction not found for {video_id}")
        
        all_gt_captions.append(gt_text)
        all_pred_captions.append(pred_text)
        processing_status.append(status)
    
    results['processing_status'] = processing_status
    
    # Evaluate with Local CLAIR
    evaluation_results = evaluate_with_local_clair(
        all_gt_captions, all_pred_captions, video_ids, 
        clair_evaluator, args.max_caption_length
    )
    
    results['clair_evaluation'] = evaluation_results
    
    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")
    
    # Create summary file
    summary_path = args.output_file.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LOCAL CLAIR EVALUATION DETAILED SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write("-"*30 + "\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"GT Directory: {args.gt_caption_dir}\n")
        f.write(f"Pred Directory: {args.pred_caption_dir}\n")
        f.write(f"Model: {args.sentence_model}\n")
        f.write(f"Device: {results['configuration']['device']}\n")
        f.write(f"Max Caption Length: {args.max_caption_length} words\n\n")
        
        if 'aggregate_statistics' in evaluation_results:
            stats = evaluation_results['aggregate_statistics']
            f.write("Overall Statistics:\n")
            f.write("-"*30 + "\n")
            f.write(f"Videos Evaluated: {evaluation_results['num_evaluated']}/{evaluation_results['num_total']}\n")
            if 'mean_score' in stats:
                f.write(f"Mean Score: {stats['mean_score']:.4f} ± {stats['std_score']:.4f}\n")
                f.write(f"Median Score: {stats['median_score']:.4f}\n")
                f.write(f"Range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]\n\n")
                
                f.write("Component-wise Statistics:\n")
                f.write("-"*30 + "\n")
                if 'component_means' in stats:
                    for comp, mean_val in stats['component_means'].items():
                        if comp != 'final_score':
                            std_val = stats['component_stds'].get(comp, 0)
                            f.write(f"{comp:<25}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Per-Video Results:\n")
        f.write("="*70 + "\n")
        
        for video_id in video_ids:
            if video_id in evaluation_results['per_video_results']:
                result = evaluation_results['per_video_results'][video_id]
                f.write(f"\n{video_id}:\n")
                f.write(f"  Status: {result['status']}\n")
                
                if result['status'] == 'success':
                    f.write(f"  CLAIR Score: {result['score']:.4f}\n")
                    f.write(f"  GT Length: {result['gt_length']} words\n")
                    f.write(f"  Pred Length: {result['pred_length']} words\n")
                    if 'components' in result:
                        f.write("  Components:\n")
                        for comp, val in result['components'].items():
                            if comp != 'final_score':
                                f.write(f"    {comp:<25}: {val:.4f}\n")
                elif result['status'] == 'skipped':
                    f.write(f"  Reason: {result.get('reason', 'unknown')}\n")
                elif result['status'] == 'error':
                    f.write(f"  Error: {result.get('error', 'unknown')}\n")
    
    print(f"Summary saved to: {summary_path}")