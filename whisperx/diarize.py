import numpy as np
import pandas as pd
import re
import torch
from pyannote.audio import Pipeline
from typing import Optional, Union, Dict, List
from transformers import pipeline
import logging

from .audio import load_audio, SAMPLE_RATE
from .types import AlignedTranscriptionResult, TranscriptionResult

class NameExtractor:
    def __init__(self):
        # Using a more appropriate NER model for name extraction
        self.ner_pipeline = pipeline(
            "ner", 
            model="dslim/bert-large-NER",  # Larger model for better accuracy
            aggregation_strategy="simple"   # Better handling of multi-token entities
        )
        self.logger = logging.getLogger(__name__)
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize extracted names"""
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s]', '', name).strip()
        # Capitalize properly
        return ' '.join([part.capitalize() for part in name.split()])
    
    def _validate_name(self, name: str) -> bool:
        """Basic validation that the extracted text looks like a name"""
        if not name:
            return False
        # Should contain at least one space (first and last name)
        if ' ' not in name:
            return False
        # Should be at least 3 characters per name part
        parts = name.split()
        if any(len(part) < 2 for part in parts):
            return False
        return True
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extracts the most likely name from text using NER with better post-processing"""
        try:
            entities = self.ner_pipeline(text)
            person_entities = [e for e in entities if e['entity_group'] == 'PER']
            
            if person_entities:
                # Get the longest PER entity (most likely to be a full name)
                longest_entity = max(person_entities, key=lambda x: x['end'] - x['start'])
                name = self._clean_name(longest_entity['word'])
                if self._validate_name(name):
                    return name
                
            # Fallback to pattern matching if NER fails
            patterns = [
                r"(?:my name is|i'm|i am|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:here|speaking)",
                r"(?:called|named)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    name = self._clean_name(match.group(1))
                    if self._validate_name(name):
                        return name
                        
        except Exception as e:
            self.logger.error(f"Error extracting name: {str(e)}")
            
        return None


class DiarizationPipeline:
    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)
        self.name_extractor = NameExtractor()
        self.logger = logging.getLogger(__name__)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio_data = {
                'waveform': torch.from_numpy(audio[None, :]),
                'sample_rate': SAMPLE_RATE
            }
            segments = self.model(audio_data, num_speakers=num_speakers, 
                                min_speakers=min_speakers, max_speakers=max_speakers)
            diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), 
                                    columns=['segment', 'label', 'speaker'])
            diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
            diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
            return diarize_df
        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            raise


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    fill_nearest: bool = False,
    extract_names: bool = True,
    num_lines_to_check: int = 2,
    min_words_for_name: int = 3
) -> dict:
    # First assign speakers without names
    transcript_result = _assign_speakers_basic(diarize_df, transcript_result, fill_nearest)
    
    if extract_names:
        # Extract and apply speaker names
        speaker_name_map = _extract_speaker_names(
            diarize_df, 
            transcript_result,
            num_lines_to_check,
            min_words_for_name
        )
        _apply_speaker_names(diarize_df, transcript_result, speaker_name_map)
    
    return transcript_result


def _assign_speakers_basic(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    fill_nearest: bool
) -> dict:
    # Pre-calculate intersections for all segments at once
    transcript_segments = transcript_result["segments"]
    
    for seg in transcript_segments:
        seg_start = seg['start']
        seg_end = seg['end']
        
        # Calculate intersections with all diarization segments
        intersections = np.minimum(diarize_df['end'], seg_end) - np.maximum(diarize_df['start'], seg_start)
        intersections[intersections < 0] = 0
        
        if not fill_nearest:
            valid = intersections > 0
            dia_tmp = diarize_df[valid]
            intersections = intersections[valid]
        else:
            dia_tmp = diarize_df
        
        if len(dia_tmp) > 0:
            # Find speaker with maximum overlap
            speaker = dia_tmp.iloc[np.argmax(intersections)]['speaker']
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    word_start = word['start']
                    word_end = word['end']
                    
                    word_intersections = np.minimum(diarize_df['end'], word_end) - np.maximum(diarize_df['start'], word_start)
                    word_intersections[word_intersections < 0] = 0
                    
                    if not fill_nearest:
                        valid = word_intersections > 0
                        dia_tmp = diarize_df[valid]
                        word_intersections = word_intersections[valid]
                    else:
                        dia_tmp = diarize_df
                    
                    if len(dia_tmp) > 0:
                        speaker = dia_tmp.iloc[np.argmax(word_intersections)]['speaker']
                        word["speaker"] = speaker
    return transcript_result


def _extract_speaker_names(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    num_lines_to_check: int,
    min_words_for_name: int
) -> Dict[str, str]:
    name_extractor = NameExtractor()
    speaker_name_map = {}
    processed_speakers = set()
    
    for seg in transcript_result["segments"]:
        speaker = seg.get("speaker")
        if not speaker or speaker in processed_speakers:
            continue
            
        # Get first few segments for this speaker
        speaker_segments = [s for s in transcript_result["segments"] 
                          if s.get("speaker") == speaker][:num_lines_to_check]
        
        # Combine text with proper spacing
        combined_text = " ".join(s["text"].strip() for s in speaker_segments if "text" in s)
        
        # Only try to extract name if we have enough words
        if len(combined_text.split()) >= min_words_for_name:
            name = name_extractor.extract_name(combined_text)
            if name:
                speaker_name_map[speaker] = name
                processed_speakers.add(speaker)
    
    return speaker_name_map


def _apply_speaker_names(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_name_map: Dict[str, str]
):
    # First preserve original speaker IDs
    for seg in transcript_result["segments"]:
        if "speaker" in seg:
            seg["original_speaker_id"] = seg["speaker"]
    
    # Replace in diarization DataFrame
    diarize_df["speaker"] = diarize_df["speaker"].apply(
        lambda x: speaker_name_map.get(x, x)
    )
    
    # Replace in transcript segments
    for seg in transcript_result["segments"]:
        if "speaker" in seg:
            seg["speaker"] = speaker_name_map.get(seg["speaker"], seg["speaker"])
        
        # Replace in words if present
        if "words" in seg:
            for word in seg["words"]:
                if "speaker" in word:
                    word["speaker"] = speaker_name_map.get(word["speaker"], word["speaker"])


class Segment:
    def __init__(self, start: int, end: int, speaker: Optional[str] = None):
        self.start = start
        self.end = end
        self.speaker = speaker


def process_audio_with_speaker_names(
    audio_path: str,
    transcript_result: TranscriptionResult,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    fill_nearest: bool = False,
    num_lines_to_check: int = 2,
    min_words_for_name: int = 3
) -> tuple[pd.DataFrame, dict, Dict[str, str]]:
    """
    Complete processing pipeline with speaker name extraction.
    
    Returns:
        - diarization DataFrame
        - updated transcript result
        - speaker name mapping dictionary
    """
    # Run diarization
    diarize_pipeline = DiarizationPipeline()
    diarize_df = diarize_pipeline(
        audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    # Assign speakers and extract names
    updated_transcript = assign_word_speakers(
        diarize_df,
        transcript_result,
        fill_nearest=fill_nearest,
        extract_names=True,
        num_lines_to_check=num_lines_to_check,
        min_words_for_name=min_words_for_name
    )
    
    # Get the final speaker name mapping
    speaker_name_map = {}
    for seg in updated_transcript["segments"]:
        if "speaker" in seg and isinstance(seg["speaker"], str):
            original_id = seg.get("original_speaker_id", seg["speaker"])
            speaker_name_map[original_id] = seg["speaker"]
    
    return diarize_df, updated_transcript, speaker_name_map