import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from finvader import finvader
from huggingface_hub import hf_hub_download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from exorde.models import (
    Classification,
    LanguageScore,
    Sentiment,
    Embedding,
    TextType,   
    Emotion,
    Irony,
    Age,
    Gender,
    Analysis,
)

logging.basicConfig(level=logging.INFO)

def initialize_models(device):
    logging.info("[TAGGING] Initializing models to be pre-ready for batch processing:")
    models = {}
    
    logging.info("[TAGGING] Loading model: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")
    models['zs_pipe'] = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        device=device
    )
    logging.info("[TAGGING] Loading model: sentence-transformers/all-MiniLM-L6-v2")
    models['sentence_transformer'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    text_classification_models = [
        ("Emotion", "SamLowe/roberta-base-go_emotions"),
        ("Irony", "cardiffnlp/twitter-roberta-base-irony"),
        ("TextType", "marieke93/MiniLM-evidence-types"),
    ]
    for col_name, model_name in text_classification_models:
        logging.info(f"[TAGGING] Loading model: {model_name}")
        models[col_name] = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
            max_length=512,
            padding=True,
        )
    
    logging.info("[TAGGING] Loading model: bert-large-uncased")
    models['bert_tokenizer'] = AutoTokenizer.from_pretrained("bert-large-uncased")
    logging.info("[TAGGING] Loading model: vaderSentiment")
    models['sentiment_analyzer'] = SentimentIntensityAnalyzer()
    try:
        emoji_lexicon = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection",
            filename="emoji_unic_lexicon.json",
        )
        loughran_dict = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection", filename="loughran_dict.json"
        )
        logging.info("[TAGGING] Loading Loughran_dict & unic_emoji_dict for sentiment_analyzer.")
        with open(emoji_lexicon) as f:
            unic_emoji_dict = json.load(f)
        with open(loughran_dict) as f:
            Loughran_dict = json.load(f)
        models['sentiment_analyzer'].lexicon.update(Loughran_dict)
        models['sentiment_analyzer'].lexicon.update(unic_emoji_dict)
    except Exception as e:
        logging.info("[TAGGING] Error loading Loughran_dict & unic_emoji_dict for sentiment_analyzer. Doing without.")
    
    logging.info("[TAGGING] Loading model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_tokenizer'] = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    logging.info("[TAGGING] Loading model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_model'] = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models['fdb_pipe'] = pipeline(
        "text-classification",
        model=models['fdb_model'],
        tokenizer=models['fdb_tokenizer'],
        top_k=None, 
        max_length=512,
        padding=True,
    )
    
    logging.info("[TAGGING] Loading model: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_tokenizer'] = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    logging.info("[TAGGING] Loading model: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_model'] = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models['gdb_pipe'] = pipeline(
        "text-classification",
        model=models['gdb_model'],
        tokenizer=models['gdb_tokenizer'],
        top_k=None, 
        max_length=512,
        padding=True,
    )
    logging.info("[TAGGING] Models loaded successfully.")
    
    return models

def batch_sentiment_analysis(documents: list[str], models: dict) -> tuple[list[float], list[float], list[float], list[float]]:
    """Optimized batch sentiment analysis"""
    sentiment_analyzer = models['sentiment_analyzer']
    fdb_pipe = models['fdb_pipe']
    gdb_pipe = models['gdb_pipe']
    
    # Batch process sentiment models
    fdb_predictions = fdb_pipe(documents)
    gdb_predictions = gdb_pipe(documents)
    
    # Batch process VADER sentiment
    vader_scores = [round(sentiment_analyzer.polarity_scores(text)["compound"], 2) for text in documents]
    
    # Batch process FinVADER sentiment  
    fin_vader_scores = [round(finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound'), 2) for text in documents]
    
    # Process FDB sentiment scores
    fdb_scores = []
    for prediction in fdb_predictions:
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else {}
        fdb_sentiment_dict = {e["label"]: round(e["score"], 3) for e in prediction}
        fdb_scores.append(round(fdb_sentiment_dict.get("positive", 0) - fdb_sentiment_dict.get("negative", 0), 3))
    
    # Process GDB sentiment scores
    gdb_scores = []
    for prediction in gdb_predictions:
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else {}
        gen_distilbert_sent = {e["label"]: round(e["score"], 3) for e in prediction}
        gdb_scores.append(round(gen_distilbert_sent.get("positive", 0) - gen_distilbert_sent.get("negative", 0), 3))
    
    return vader_scores, fin_vader_scores, fdb_scores, gdb_scores


def compute_compound_sentiments(vader_scores: list[float], fin_vader_scores: list[float], 
                               fdb_scores: list[float], gdb_scores: list[float]) -> tuple[list[float], list[float]]:
    """Compute compound sentiments efficiently"""
    compound_financial_sentiments = []
    compound_sentiments = []
    
    for i in range(len(vader_scores)):
        # Compound financial sentiment
        fin_compound = round((0.70 * fdb_scores[i] + 0.30 * fin_vader_scores[i]), 2)
        compound_financial_sentiments.append(fin_compound)
        
        # Compound general sentiment
        if abs(fin_compound) >= 0.6:
            compound = round((0.30 * gdb_scores[i] + 0.10 * vader_scores[i] + 0.60 * fin_compound), 2)
        elif abs(fin_compound) >= 0.4:
            compound = round((0.40 * gdb_scores[i] + 0.20 * vader_scores[i] + 0.40 * fin_compound), 2)
        elif abs(fin_compound) >= 0.1:
            compound = round((0.60 * gdb_scores[i] + 0.25 * vader_scores[i] + 0.15 * fin_compound), 2)
        else:
            compound = round((0.60 * gdb_scores[i] + 0.40 * vader_scores[i]), 2)
        
        compound_sentiments.append(compound)
    
    return compound_sentiments, compound_financial_sentiments


def tag(documents: list[str], lab_configuration):
    """Optimized batch tagging function"""
    # Validate inputs
    for doc in documents:
        assert isinstance(doc, str)
    
    if not documents:
        return []
    
    # Loading models from lab configuration
    models = lab_configuration["models"]
    
    logging.info(f"Starting Optimized Tagging Batch pipeline for {len(documents)} documents...")
    
    # 1. BATCH EMBEDDING GENERATION
    logging.info("Processing embeddings...")
    sentence_transformer = models['sentence_transformer']
    embeddings = sentence_transformer.encode(documents, convert_to_numpy=True, show_progress_bar=False)
    embeddings_list = [list(emb.astype(float)) for emb in embeddings]
    
    # 2. BATCH CLASSIFICATION
    logging.info("Processing classifications...")
    zs_pipe = models['zs_pipe']
    classification_labels = list(lab_configuration["labeldict"].keys())
    classifications = zs_pipe(documents, candidate_labels=classification_labels)
    
    # 3. BATCH TEXT CLASSIFICATION (Emotion, Irony, TextType)
    logging.info("Processing text classifications...")
    text_classification_results = {}
    text_classification_models = ["Emotion", "Irony", "TextType"]
    
    # Process each model in batch
    for col_name in text_classification_models:
        pipe = models[col_name]
        predictions = pipe(documents)
        # Convert to expected format
        text_classification_results[col_name] = [
            [(y["label"], float(y["score"])) for y in pred] if isinstance(pred, list) else [(pred["label"], float(pred["score"]))]
            for pred in predictions
        ]
    
    # 4. BATCH SENTIMENT ANALYSIS
    logging.info("Processing sentiment analysis...")
    vader_scores, fin_vader_scores, fdb_scores, gdb_scores = batch_sentiment_analysis(documents, models)
    compound_sentiments, compound_financial_sentiments = compute_compound_sentiments(
        vader_scores, fin_vader_scores, fdb_scores, gdb_scores
    )
    
    # 5. BUILD RESULTS
    logging.info("Building analysis results...")
    results = []
    
    for i in range(len(documents)):
        # Create sentiment
        sentiment = Sentiment(compound_sentiments[i])
        
        # Create embedding
        embedding = Embedding(embeddings_list[i])
        
        # Create classification
        classification_result = classifications[i] if isinstance(classifications, list) else classifications
        top_label = classification_result["labels"][0]
        top_score = round(classification_result["scores"][0], 4)
        classification = Classification(label=top_label, score=top_score)
        
        # Mock gender (as in original)
        gender = Gender(male=0.5, female=0.5)
        
        # Text type
        types = {item[0]: item[1] for item in text_classification_results["TextType"][i]}
        text_type = TextType(
            assumption=types.get("Assumption", 0.0),
            anecdote=types.get("Anecdote", 0.0),
            none=types.get("None", 0.0),
            definition=types.get("Definition", 0.0),
            testimony=types.get("Testimony", 0.0),
            other=types.get("Other", 0.0),
            study=types.get("Statistics/Study", 0.0),
        )
        
        # Emotions
        emotions = {item[0]: item[1] for item in text_classification_results["Emotion"][i]}
        # Round all values to 4 decimal places
        emotions = {k: round(v, 4) for k, v in emotions.items()}
        emotion = Emotion(
            love=emotions.get("love", 0.0),
            admiration=emotions.get("admiration", 0.0),
            joy=emotions.get("joy", 0.0),
            approval=emotions.get("approval", 0.0),
            caring=emotions.get("caring", 0.0),
            excitement=emotions.get("excitement", 0.0),
            gratitude=emotions.get("gratitude", 0.0),
            desire=emotions.get("desire", 0.0),
            anger=emotions.get("anger", 0.0),
            optimism=emotions.get("optimism", 0.0),
            disapproval=emotions.get("disapproval", 0.0),
            grief=emotions.get("grief", 0.0),
            annoyance=emotions.get("annoyance", 0.0),
            pride=emotions.get("pride", 0.0),
            curiosity=emotions.get("curiosity", 0.0),
            neutral=emotions.get("neutral", 0.0),
            disgust=emotions.get("disgust", 0.0),
            disappointment=emotions.get("disappointment", 0.0),
            realization=emotions.get("realization", 0.0),
            fear=emotions.get("fear", 0.0),
            relief=emotions.get("relief", 0.0),
            confusion=emotions.get("confusion", 0.0),
            remorse=emotions.get("remorse", 0.0),
            embarrassment=emotions.get("embarrassment", 0.0),
            surprise=emotions.get("surprise", 0.0),
            sadness=emotions.get("sadness", 0.0),
            nervousness=emotions.get("nervousness", 0.0),
        )
        
        # Irony
        ironies = {item[0]: item[1] for item in text_classification_results["Irony"][i]}
        irony = Irony(irony=ironies.get("irony", 0.0), non_irony=ironies.get("non_irony", 0.0))
        
        # Age (untrained model)
        age = Age(below_twenty=0.0, twenty_thirty=0.0, thirty_forty=0.0, forty_more=0.0)
        
        # Language score (untrained model)
        language_score = LanguageScore(1.0)  # default value
        
        # Create analysis object
        analysis = Analysis(
            classification=classification,
            language_score=language_score,
            sentiment=sentiment,
            embedding=embedding,
            gender=gender,
            text_type=text_type,
            emotion=emotion,
            irony=irony,
            age=age,
        )
        
        results.append(analysis)
    
    logging.info(f"Completed optimized tagging for {len(results)} documents")
    return results
