# Unmodified helpers from HUST-AI-HYZ/MemoryAgentBench commit 538026089d1a8a8eff05121d0db89b388f360eba.
# Copyright (c) 2026 Yuanzhe Hu; MIT license in LICENSE.
# Used only by parity tests, not the benchmark runtime.
import re
import string
from collections import Counter
import numpy as np
import nltk
import tiktoken
from nltk.metrics.distance import edit_distance

def normalize_answer(answer_text):
    """
    Normalize text for evaluation by removing articles, punctuation, and extra whitespace.
    
    Args:
        answer_text: The text to normalize
        
    Returns:
        Normalized text string
    """
    # Apply all normalization steps in sequence
    text = answer_text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    
    return text

def f1_score(prediction, ground_truth):
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # Handle special cases for yes/no/noanswer responses
    special_answers = {'yes', 'no', 'noanswer'}
    if ((normalized_prediction in special_answers or normalized_ground_truth in special_answers) and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    # Tokenize both texts and calculate token overlap
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common_tokens = sum(common_tokens.values())
    
    if num_common_tokens == 0:
        return ZERO_METRIC
    
    # Calculate precision, recall, and F1
    precision = num_common_tokens / len(prediction_tokens)
    recall = num_common_tokens / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall

def drqa_exact_match_score(prediction, ground_truth):
    """
    Check if prediction is an exact match with ground truth after normalization.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating exact match
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def substring_exact_match_score(prediction, ground_truth):
    """
    Check if ground truth is a substring of the prediction after normalization.
    
    Args:
        prediction: The predicted text  
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating substring match
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)

def drqa_metric_max_over_ground_truths(metric_function, prediction, ground_truths):
    """
    Calculate the maximum score over multiple ground truth answers.
    
    Args:
        metric_function: Function to calculate score between prediction and single ground truth
        prediction: The predicted text
        ground_truths: List of ground truth answers (can be string, list, or nested list)
        
    Returns:
        Maximum score across all ground truths
    """
    # Normalize ground_truths to a flat list of strings
    if isinstance(ground_truths, str):
        ground_truth_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        # Flatten nested lists
        ground_truth_list = [gt for gt_sublist in ground_truths for gt in gt_sublist]
    else:
        ground_truth_list = ground_truths

    # Calculate score for each ground truth and return maximum
    return max(metric_function(prediction, gt) for gt in ground_truth_list)

def parse_output(output_text, answer_prefix="Answer:"):
    """
    Parse model output to extract the answer portion.
    
    Args:
        output_text: The complete model output
        answer_prefix: The prefix that indicates where the answer starts
        
    Returns:
        Extracted answer text or None if not found
    """
    # Try multiple patterns to extract the answer
    extraction_patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), 
        re.compile(r"(?:^)(.*)(?:\n|$)")
    ]
    
    for pattern in extraction_patterns:
        match = pattern.search(output_text)
        if match:
            extracted_text = match[1].strip()
            # Remove prefix again in case it was repeated
            clean_answer = re.sub(f'^{re.escape(answer_prefix)}', '', extracted_text, flags=re.IGNORECASE).strip()
            return clean_answer
    
    # Should rarely reach here, but return None if no pattern matches
    return None

def chunk_text_into_sentences(text, model_name="gpt-4o-mini", chunk_size=4096):
    """
    Split text into chunks of specified token size, preserving sentence boundaries.
    
    Args:
        text: The long text document to be split
        model_name: The tokenizer model name (default: gpt-4o-mini)
        chunk_size: Maximum number of tokens allowed per chunk
        
    Returns:
        List of text chunks, each within the specified token limit
    """
    # Ensure NLTK sentence tokenizer is available
    nltk.download('punkt', quiet=True)
    
    # Initialize tokenizer with fallback
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Use fallback model if specified model is not recognized
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    text_chunks = []
    current_chunk_sentences = []
    current_chunk_token_count = 0

    for sentence in sentences:
        # Count tokens in current sentence
        sentence_tokens = encoding.encode(sentence, allowed_special={'<|endoftext|>'})
        sentence_token_count = len(sentence_tokens)
        
        # Check if adding this sentence would exceed chunk size
        if current_chunk_token_count + sentence_token_count > chunk_size:
            # Finalize current chunk and start new one
            text_chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_token_count += sentence_token_count
    
    # Add final chunk if it contains any sentences
    if current_chunk_sentences:
        text_chunks.append(" ".join(current_chunk_sentences))
    
    return text_chunks

def clean_text_elements(text, remove_parentheses=True, normalize_ws=True, remove_nums=True):
    """Clean text by removing various elements."""
    if remove_parentheses:
        text = re.sub(r"\([^()]*\)", "", text)
    if remove_nums:
        text = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)
    if normalize_ws:
        text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_parentheses(text):
    """Remove content within parentheses from text."""
    return re.sub(r"\([^()]*\)", "", text)

def normalize_whitespace(text):
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()

def remove_numbering(text):
    """Remove numbering from the beginning of text."""
    return re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)

def extract_movie_name(text):
    """
    Extract and clean movie name from file path or text.
    
    Args:
        text: Raw text containing movie name
        
    Returns:
        Cleaned movie name
    """
    # Extract filename if it's a path
    filename = text.split('/')[-1]
    # Replace common separators with spaces
    cleaned_name = filename.replace('_', ' ').replace('-', ' ').replace('>', ' ')
    # Apply cleaning functions
    return normalize_whitespace(clean_parentheses(cleaned_name))

def find_nearest_movie(target_name, candidate_movies):
    """
    Find the nearest movie name using edit distance.
    
    Args:
        target_name: The movie name to match
        candidate_movies: List of candidate movie names
        
    Returns:
        Dictionary with matching information
    """
    # Remove duplicates from candidates
    unique_candidates = list(set(candidate_movies))
    
    # Calculate edit distances
    distances = [edit_distance(target_name.lower(), candidate.lower()) 
                for candidate in unique_candidates]
    
    # Find nearest match
    nearest_index = np.argmin(distances)
    nearest_movie = unique_candidates[nearest_index]
    
    return {
        'movie_name': target_name, 
        'min_edit_distance': distances[nearest_index], 
        'nearest_movie': nearest_movie
    }

def extract_recommendation_list(text, movie_candidates=None):
    """
    Extract recommendation list from text output.
    
    Args:
        text: Text containing recommendations
        movie_candidates: Optional list of valid movie names for matching
        
    Returns:
        Tuple of (recommendation_list, preference_text)
    """
    try:
        # Try to split on first numbered item
        preference_text, recommendation_text = text.split('1.', maxsplit=1)
    except Exception as e:
        print(e)
        preference_text = ""
        # Fallback: replace commas with newlines for parsing
        recommendation_text = text.replace(',', '\n')
    
    # Extract and clean recommendation items using the consolidated function
    raw_recommendations = [
        clean_text_elements(item.strip()) for item in recommendation_text.split('\n')
    ]
    
    # Match against candidates if provided
    recommendation_list = ([find_nearest_movie(item, movie_candidates) for item in raw_recommendations] 
                         if movie_candidates is not None else raw_recommendations)
    
    return recommendation_list, preference_text
