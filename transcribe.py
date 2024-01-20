import argparse
from typing import Dict

import nemo.collections.asr as nemo_asr


def predict_model(
        model_path: str = None,
        audio_file_path: str = None
    ) -> Dict:
    # Restore the ASR model from the provided path
    model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
    # Transcribe the given audio file
    text = model.transcribe([audio_file_path])
    print({"result": text[0]})


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, help="Path to a model to evaluate.")
    parser.add_argument("--audio_file_path", default=None, help="Path for train manifest JSON file.")
    args = parser.parse_args()

    predict_model(model_path=args.model_path, audio_file_path=args.audio_file_path)
