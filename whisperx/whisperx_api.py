import gc
import os
import warnings
import torch
from typing import Optional, List, Union
from .types import AlignedTranscriptionResult, TranscriptionResult

class WhisperXTranscriber:
    def __init__(self, 
                 model_name: str = "small",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 compute_type: str = "float16",
                 hf_token: Optional[str] = None,
                 model_dir: Optional[str] = None):
        """
        Initialize the WhisperX transcriber with common settings.
        
        Args:
            model_name: Name of Whisper model (small, medium, large, etc.)
            device: Computation device (cuda/cpu)
            compute_type: Computation precision (float16/float32)
            hf_token: HuggingFace token for diarization
            model_dir: Directory to cache models
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.hf_token = hf_token
        self.model_dir = model_dir
        self._model = None
        self._align_model = None
        self._diarize_model = None

        # Configure torch for better reproducibility
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def transcribe(
        self,
        audio_path: Union[str, List[str]],
        language: str = "en",
        diarize: bool = False,
        align: bool = True,
        vad_options: Optional[dict] = None,
        asr_options: Optional[dict] = None,
        output_dir: str = ".",
        output_format: str = "all",
        **kwargs
    ) -> dict:
        """
        Main transcription method.
        
        Args:
            audio_path: Path to audio file or list of paths
            language: Language code (en, fr, etc.)
            diarize: Whether to perform speaker diarization
            align: Whether to perform word alignment
            vad_options: Dictionary of VAD options
            asr_options: Dictionary of ASR options
            output_dir: Output directory
            output_format: Output format (all, json, srt, etc.)
            **kwargs: Additional writer options
            
        Returns:
            Dictionary with transcription results and metadata
        """
        # Set default options
        vad_options = vad_options or {
            "vad_method": "pyannote",
            "vad_onset": 0.500,
            "vad_offset": 0.363,
            "chunk_size": 30
        }
        
        asr_options = asr_options or {
            "beam_size": 5,
            "patience": 1.0,
            "length_penalty": 1.0,
            "temperatures": [0],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "suppress_tokens": [-1],
            "suppress_numerals": False
        }

        # Load required modules
        from .asr import load_model
        from .audio import load_audio
        from .alignment import align, load_align_model
        from .diarize import DiarizationPipeline, assign_word_speakers
        from .utils import get_writer

        # Load ASR model
        self._model = load_model(
            self.model_name,
            device=self.device,
            download_root=self.model_dir,
            compute_type=self.compute_type,
            language=language,
            asr_options=asr_options,
            vad_method=vad_options["vad_method"],
            vad_options=vad_options,
            local_files_only=False
        )

        # Process audio files
        if isinstance(audio_path, str):
            audio_path = [audio_path]

        results = []
        for path in audio_path:
            audio = load_audio(path)
            print(f">> Processing {os.path.basename(path)}...")
            
            # Transcription
            result = self._model.transcribe(audio, batch_size=8)
            
            # Ensure language is set in the result
            if 'language' not in result:
                result['language'] = language
            elif language != result['language']:
                print(f"Warning: Detected language {result['language']} differs from specified language {language}")
                language = result['language']  # Use detected language
            
            # Alignment
            if align and not (language == "en" and self.model_name.endswith(".en")):
                print(">> Aligning transcript...")
                self._align_model, align_metadata = load_align_model(
                    language, self.device
                )
                if len(result["segments"]) > 0:
                    result = align(
                        result["segments"],
                        self._align_model,
                        align_metadata,
                        path,
                        self.device
                    )
                    result["language"] = language  # Ensure language is preserved
            
            # Diarization
            if diarize:
                if not self._diarize_model:
                    print(">> Initializing diarization pipeline...")
                    self._diarize_model = DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                print(">> Performing speaker diarization...")
                diarize_segments = self._diarize_model(
                    path,
                    min_speakers=kwargs.get("min_speakers", 1),
                    max_speakers=kwargs.get("max_speakers", 2)
                )
                result = assign_word_speakers(diarize_segments, result)
                result["language"] = language  # Ensure language is preserved
            
            # Prepare writer options
            writer_args = {
                "highlight_words": kwargs.get("highlight_words", False),
                "max_line_count": kwargs.get("max_line_count", None),
                "max_line_width": kwargs.get("max_line_width", None),
                "segment_resolution": kwargs.get("segment_resolution", "sentence")
            }
            
            # Write output
            os.makedirs(output_dir, exist_ok=True)
            writer = get_writer(output_format, output_dir)
            writer(result, path, writer_args)
            
            results.append({
                "audio_path": path,
                "result": result,
                "output_dir": output_dir
            })
        
        return results

    def cleanup(self):
        """Clean up models and free memory"""
        if self._model:
            del self._model
        if self._align_model:
            del self._align_model
        if self._diarize_model:
            del self._diarize_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()