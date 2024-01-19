import argparse
from typing import Dict

import nemo.collections.asr as nemo_asr
import torch
from omegaconf import open_dict


def evaluate_model(model_path: str, test_manifest: str, batch_size: int = 1) -> Dict:

    # Determine the device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Restore the ASR model from the provided path
    model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
    model.to(device)
    model.eval()

    # Update the model configuration for evaluation
    with open_dict(model.cfg):
        model.cfg.validation_ds.manifest_filepath = test_manifest
        model.cfg.validation_ds.batch_size = batch_size

    # Set up the test data using the updated configuration
    model.setup_test_data(model.cfg.validation_ds)

    wer_nums = []
    wer_denoms = []

    # Iterate through the test data
    for test_batch in model.test_dataloader():
        # Extract elements from the test batch
        test_batch = [x for x in test_batch]
        targets = test_batch[2].to(device)
        targets_lengths = test_batch[3].to(device)
        # Forward pass through the model
        log_probs, encoded_len, greedy_predictions = model(input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device))
        # Compute Word Error Rate (WER) and store results
        model._wer.update(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = model._wer.compute()
        model._wer.reset()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())
        # Free up memory by deleting variables
        del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

    # Compute the WER score
    wer_score = sum(wer_nums) / sum(wer_denoms)
    print({"WER_score": wer_score})


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, help="Path to a model to evaluate.")
    parser.add_argument("--test_manifest", help="Path for train manifest JSON file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the dataset to train.")
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path, test_manifest=args.test_manifest, batch_size=args.batch_size)
