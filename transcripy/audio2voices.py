from pyannote.audio import Pipeline
import os
from typing import Any, TypedDict, Tuple
import torch
import logging

from .helper import MultiFileHandler, read_rttm

class DiarizationTurn(TypedDict):
    start: float
    end: float

DiarizationTrack = Tuple[DiarizationTurn, Any, str]

class MultiDetector(MultiFileHandler):
    def __init__(self, data_path: str, verbose: bool = False, model: str = "pyannote/speaker-diarization@2.1") -> None:
        super().__init__(data_path, verbose, "raw_audio_voices",
                         "diarization", "rttm", ["wav"])
        try:
            self.pipeline = Pipeline.from_pretrained(model, use_auth_token="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")
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {str(e)}")
            logging.error("Make sure you have accepted the user conditions at https://huggingface.co/pyannote/speaker-diarization")
            self.pipeline = None

    def handler(self, input_file: str, output_file: str, file_idx: int) -> None:
        if self.pipeline is None:
            logging.error("Pipeline is not initialized. Skipping diarization.")
            return

        uri, audio = os.path.split(input_file)
        file_identifier = {'uri': audio.replace(" ", "_"), 'audio': input_file}
        try:
            diarization = self.pipeline(file_identifier)
            with open(output_file, "w") as rttm:
                diarization.write_rttm(rttm)
            _id, diarization = read_rttm(output_file)
        except Exception as e:
            logging.error(f"Error during diarization: {str(e)}")
