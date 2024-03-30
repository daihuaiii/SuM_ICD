import torch
import warnings
from typing import Callable, List, Tuple, Dict

StateType = Dict[str, torch.Tensor]
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]


class BeamSearch:

    def __init__(self,
                 end_index: int,
                 max_steps: int = 15,
                 beam_size: int = 2,
                 per_node_beam_size: int = None) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.device = torch.device("cuda")

    def search(self,
               start_predictions: torch.Tensor,
               batch_input: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType):

        batch_size = start_predictions.size()[0]

        predictions: List[torch.Tensor] = []

        merged_source_local_mask = torch.cat((batch_input["source_l_attention_mask"],
                                              batch_input["source_s_attention_mask"]), dim=-1)
        merged_source_local_mask = torch.sort(merged_source_local_mask, dim=-1, descending=True).values

        merged_source_local_mask = (merged_source_local_mask.float() + 1e-45).log().to(self.device)
        start_class_log_probabilities, state = step(start_predictions, start_state)
        start_class_log_probabilities = start_class_log_probabilities + merged_source_local_mask

        num_classes = start_class_log_probabilities.size()[1]

        if self.per_node_beam_size > num_classes:

            raise Exception(f"Target vocab size ({num_classes:d}) too small "
                            f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                            f"Please decrease beam_size or per_node_beam_size.")

        start_top_log_probabilities, start_predicted_classes = start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities
        last_log_probabilities = start_top_log_probabilities

        predictions.append(start_predicted_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes),
            float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.

        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            state[key] = state_tensor.unsqueeze(1).expand(batch_size, self.beam_size, *last_dims)\
                .reshape(batch_size * self.beam_size, *last_dims)

        backpointers = list()
        for timestep in range(self.max_steps - 3):
            last_predictions = predictions[-1]
            if (last_predictions == self._end_index).all():
                break

            merged_index = batch_input["merged_index"].to(self.device)
            last_predicted_global_ids = merged_index.gather(-1, last_predictions.to(self.device)).reshape(batch_size * self.beam_size, 1)

            class_log_probabilities, state = step(last_predicted_global_ids, state)

            last_predictions_expanded = last_predictions.reshape(batch_size * self.beam_size).unsqueeze(-1).expand(batch_size * self.beam_size, num_classes)

            cleaned_log_probabilities = torch.where(last_predictions_expanded == self._end_index, log_probs_after_end,
                                                    class_log_probabilities)

            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(self.per_node_beam_size)

            expanded_last_log_probabilities = last_log_probabilities.unsqueeze(2)\
                .expand(batch_size, self.beam_size, self.per_node_beam_size) \
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)

            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            reshaped_summed = summed_top_log_probabilities.reshape(batch_size, self.beam_size * self.per_node_beam_size)

            reshaped_predicted_classes = predicted_classes.reshape(batch_size, self.beam_size * self.per_node_beam_size)

            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)

            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices.long())

            predictions.append(restricted_predicted_classes)

            last_log_probabilities = restricted_beam_log_probs

            backpointer = torch.true_divide(restricted_beam_indices, self.per_node_beam_size)
            backpointers.append(backpointer)

            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()

                expanded_backpointer = backpointer.view(batch_size, self.beam_size, *([1] * len(last_dims))) \
                    .expand(batch_size, self.beam_size, *last_dims)

                state[key] = state_tensor.to(self.device)\
                    .reshape(batch_size, self.beam_size, *last_dims) \
                    .gather(1, expanded_backpointer.long().to(self.device)) \
                    .reshape(batch_size * self.beam_size, *last_dims)

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            cur_preds = predictions[timestep].gather(1, cur_backpointers.long()).unsqueeze(2)
            reconstructed_predictions.append(cur_preds)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers.long())

        final_preds = predictions[0].gather(1, cur_backpointers.long()).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities