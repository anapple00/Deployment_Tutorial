import copy
import json


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices
            Mask values selected in ``[0，1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, label,real_token_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        # self.token_type_ids = token_type_ids
        self.label = label
        self.real_token_len = real_token_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Seriglizes this instance to g Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) :
        """serializes this instance to g JSON string,"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"