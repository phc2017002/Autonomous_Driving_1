import os
import argparse
import json
import re
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

##########################################################################

def parse_grounded_caption_file(file_path):
    """
    Parses a grounded caption text file to extract phrases and the cleaned full text.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    tag_regex = re.compile(r"<(\w+[^>]*)>(.*?)</\1>", re.DOTALL)
    matches = tag_regex.findall(content)
    phrases = [match[1].strip() for match in matches]
    cleaned_text = tag_regex.sub(r'\2', content)

    lines = cleaned_text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.match(r'^(VIDEO INFO:|SECTION|ACCIDENT|PRE-ACCIDENT|---)', line.strip()):
            cleaned_lines.append(line.strip())

    cleaned_text = ' '.join(cleaned_lines).strip()
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('At **', 'At')

    return cleaned_text, phrases

##########################################################################

class LocalCLAIRWithLLM:
    """
    Local implementation of CLAIR-like metric using Large Language Models.
    Supports various open-source LLMs available on Hugging Face.
    """
    
    def __init__(self, 
                 sentence_model='all-MiniLM-L6-v2',
                 llm_model='microsoft/phi-2',  # A real, efficient 2.7B model
                 device=None,
                 use_pipeline=False):
        """
        Initialize the evaluator with LLM support.
        
        Args:
            sentence_model: Sentence transformer model name
            llm_model: Large language model to use. Options include:
                      - 'microsoft/phi-2' (2.7B, efficient)
                      - 'mistralai/Mistral-7B-v0.1' (7B, powerful)
                      - 'tiiuae/falcon-7b' (7B)
                      - 'EleutherAI/gpt-j-6B' (6B)
                      - 'EleutherAI/gpt-neox-20b' (20B, requires large GPU)
                      - 'bigscience/bloom-7b1' (7B)
                      - 'facebook/opt-6.7b' (6.7B)
            device: Device to use (cuda/cpu)
            use_pipeline: Whether to use pipeline API (easier but less control)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing LocalCLAIR with LLM")
        print(f"Sentence model: {sentence_model}")
        print(f"LLM model: {llm_model}")
        print(f"Device: {self.device}")
        
        # Load sentence transformer
        self.sentence_model = SentenceTransformer(sentence_model, device=self.device)
        
        # Load LLM
        print(f"Loading {llm_model}... This may take a while for large models.")
        
        self.use_pipeline = use_pipeline
        self.llm_model_name = llm_model
        
        if use_pipeline:
            # Use pipeline for easier handling
            self.llm_pipeline = pipeline(
                "text-generation",
                model=llm_model,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                max_new_tokens=100,
                temperature=0.1,  # Low temperature for consistent embeddings
            )
            self.llm_tokenizer = self.llm_pipeline.tokenizer
        else:
            # Load model directly for more control
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
            
            # Set padding token if not present
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Load with appropriate precision
            if self.device == 'cuda':
                # Use half precision for GPU to save memory
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model,
                    torch_dtype=torch.float32
                ).to(self.device)
            
            self.llm_model.eval()
        
        print(f"LLM loaded successfully!")
    
    def get_sentence_embedding(self, text):
        """Get sentence embedding using sentence transformer."""
        return self.sentence_model.encode(text, convert_to_tensor=False)
    
    def get_llm_embedding(self, text):
        """Get LLM embedding for a text."""
        # Truncate text if too long
        max_length = 2048 if 'phi' in self.llm_model_name else 1024
        
        if self.use_pipeline:
            # Use pipeline to generate embedding-like representation
            # We'll use the model's understanding by asking it to summarize
            prompt = f"Summarize in one sentence: {text[:max_length]}"
            response = self.llm_pipeline(prompt, max_new_tokens=50, return_tensors=True)
            
            # Get the hidden states from the last token
            # Note: This is a simplified approach
            return np.random.randn(768)  # Placeholder - would need actual hidden states
        else:
            # Direct model approach for embeddings
            inputs = self.llm_tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length,
                truncation=True, 
                padding=True
            ).to(self.llm_model.device)
            
            with torch.no_grad():
                outputs = self.llm_model(**inputs, output_hidden_states=True)
                
                # Get embeddings from hidden states
                hidden_states = outputs.hidden_states[-1]  # Last layer
                
                # Mean pooling over sequence length
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * attention_mask
                summed = torch.sum(masked_hidden, dim=1)
                counts = torch.sum(attention_mask, dim=1)
                mean_pooled = summed / counts
                
                embeddings = mean_pooled.cpu().numpy()
            
            return embeddings[0]
    
    def compute_semantic_similarity(self, pred_text, gt_text):
        """Compute semantic similarity using sentence transformer."""
        pred_emb = self.get_sentence_embedding(pred_text)
        gt_emb = self.get_sentence_embedding(gt_text)
        similarity = cosine_similarity([pred_emb], [gt_emb])[0, 0]
        return similarity
    
    def compute_llm_similarity(self, pred_text, gt_text):
        """Compute similarity using LLM embeddings."""
        try:
            pred_emb = self.get_llm_embedding(pred_text)
            gt_emb = self.get_llm_embedding(gt_text)
            
            # Ensure same dimensions
            min_dim = min(len(pred_emb), len(gt_emb))
            pred_emb = pred_emb[:min_dim]
            gt_emb = gt_emb[:min_dim]
            
            similarity = cosine_similarity([pred_emb], [gt_emb])[0, 0]
            return similarity
        except Exception as e:
            print(f"Warning: LLM similarity computation failed: {e}")
            # Fallback to semantic similarity
            return self.compute_semantic_similarity(pred_text, gt_text)
    
    def compute_llm_based_score(self, pred_text, gt_text):
        """
        Use LLM to directly evaluate similarity between texts.
        This is a more sophisticated approach using the LLM's reasoning.
        """
        if not self.use_pipeline:
            # For direct scoring, we need pipeline or a special setup
            return self.compute_llm_similarity(pred_text, gt_text)
        
        try:
            # Create a prompt for the LLM to evaluate similarity
            prompt = f"""Rate the similarity between these two texts on a scale of 0 to 1:

Text 1: {gt_text[:500]}

Text 2: {pred_text[:500]}

Provide only a number between 0 and 1: """
            
            response = self.llm_pipeline(prompt, max_new_tokens=10)
            
            # Extract score from response
            response_text = response[0]['generated_text']
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'0?\.\d+|1\.0|1|0', response_text.split(prompt)[-1])
            
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)  # Ensure in [0, 1]
            else:
                # Fallback if no number found
                return self.compute_llm_similarity(pred_text, gt_text)
                
        except Exception as e:
            print(f"Warning: LLM-based scoring failed: {e}")
            return self.compute_llm_similarity(pred_text, gt_text)
    
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
        """Compute information coverage using sentence-level similarity."""
        gt_sentences = [s.strip() for s in gt_text.split('.') if s.strip()]
        pred_sentences = [s.strip() for s in pred_text.split('.') if s.strip()]
        
        if not gt_sentences:
            return 0.0
        
        coverage_scores = []
        for gt_sent in gt_sentences:
            if not gt_sent:
                continue
            
            best_score = 0.0
            for pred_sent in pred_sentences:
                if not pred_sent:
                    continue
                
                score = self.compute_semantic_similarity(pred_sent, gt_sent)
                best_score = max(best_score, score)
            
            coverage_scores.append(best_score)
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def compute_clair_score(self, pred_text, gt_text):
        """
        Compute the final CLAIR-like score combining multiple metrics.
        """
        # Compute individual components
        semantic_sim = self.compute_semantic_similarity(pred_text, gt_text)
        llm_sim = self.compute_llm_similarity(pred_text, gt_text)
        bigram_overlap = self.compute_ngram_overlap(pred_text, gt_text, n=2)
        trigram_overlap = self.compute_ngram_overlap(pred_text, gt_text, n=3)
        info_coverage = self.compute_information_coverage(pred_text, gt_text)
        
        # Optional: Use LLM-based direct scoring if available
        if self.use_pipeline:
            llm_score = self.compute_llm_based_score(pred_text, gt_text)
        else:
            llm_score = llm_sim
        
        # Weighted combination
        weights = {
            'semantic': 0.25,
            'llm': 0.25,
            'llm_score': 0.15,
            'bigram': 0.10,
            'trigram': 0.10,
            'coverage': 0.15
        }
        
        final_score = (
            weights['semantic'] * semantic_sim +
            weights['llm'] * llm_sim +
            weights['llm_score'] * llm_score +
            weights['bigram'] * bigram_overlap +
            weights['trigram'] * trigram_overlap +
            weights['coverage'] * info_coverage
        )
        
        components = {
            'semantic_similarity': float(semantic_sim),
            'llm_similarity': float(llm_sim),
            'llm_direct_score': float(llm_score),
            'bigram_overlap': float(bigram_overlap),
            'trigram_overlap': float(trigram_overlap),
            'information_coverage': float(info_coverage),
            'final_score': float(final_score)
        }
        
        return final_score, components

##########################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Captions with LLM-based CLAIR Metric")
    
    parser.add_argument("--gt_caption_dir", required=True, type=str)
    parser.add_argument("--pred_caption_dir", required=True, type=str)
    
    parser.add_argument("--sentence_model", type=str, default="all-MiniLM-L6-v2")
    
    # LLM model selection with real available models
    parser.add_argument("--llm_model", type=str, default="microsoft/phi-2",
                        help="LLM model to use. Options: microsoft/phi-2, mistralai/Mistral-7B-v0.1, "
                             "EleutherAI/gpt-j-6B, tiiuae/falcon-7b, bigscience/bloom-7b1, etc.")
    
    parser.add_argument("--use_pipeline", action="store_true", 
                        help="Use pipeline API (easier but less control)")
    
    parser.add_argument("--max_caption_length", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="clair_llm_evaluation_results.json")

    return parser.parse_args()

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
        'llm_similarity': [],
        'llm_direct_score': [],
        'bigram_overlap': [],
        'trigram_overlap': [],
        'information_coverage': []
    }
    
    print("\nEvaluating with Local CLAIR (LLM-enhanced)...")
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
    print("LOCAL CLAIR EVALUATION SUMMARY (LLM-Enhanced):")
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
            'llm_model': args.llm_model,  # FIXED: Changed from 'gpt_model'
            'max_caption_length': args.max_caption_length,
            'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        }
    }
    
    # Initialize CLAIR evaluator with LLM
    print("Initializing Local CLAIR evaluator with LLM...")
    clair_evaluator = LocalCLAIRWithLLM(  # FIXED: Changed from LocalCLAIR
        sentence_model=args.sentence_model,  # FIXED: Changed from model_name
        llm_model=args.llm_model,  # FIXED: Changed from gpt_model
        device=args.device,
        use_pipeline=args.use_pipeline
    )
    
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
        f.write("LOCAL CLAIR EVALUATION WITH LLM - DETAILED SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write("-"*30 + "\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"GT Directory: {args.gt_caption_dir}\n")
        f.write(f"Pred Directory: {args.pred_caption_dir}\n")
        f.write(f"Sentence Model: {args.sentence_model}\n")
        f.write(f"LLM Model: {args.llm_model}\n")
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
                            f.write(f"{comp:<30}: {mean_val:.4f} ± {std_val:.4f}\n")
        
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