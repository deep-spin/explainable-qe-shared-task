import torch
from transformers import BertForSequenceClassification
from collections import defaultdict


def bert_getter(model_bert, inputs_dict, forward_fn=None):

    hidden_states_ = []

    def get_hook(i):
        def hook(module, inputs, outputs=None):
            if i == 0:
                hidden_states_.append(outputs)
            elif 1 <= i <= len(model_bert.encoder.layer):
                hidden_states_.append(inputs[0])
            elif i == len(model_bert.encoder.layer) + 1:
                hidden_states_.append(outputs[0])

        return hook

    handles = (
        [model_bert.embeddings.word_embeddings.register_forward_hook(get_hook(0))]
        + [
            layer.register_forward_pre_hook(get_hook(i + 1))
            for i, layer in enumerate(model_bert.encoder.layer)
        ]
        + [
            model_bert.encoder.layer[-1].register_forward_hook(
                get_hook(len(model_bert.encoder.layer) + 1)
            )
        ]
    )

    try:
        if forward_fn is None:
            outputs = model_bert(**inputs_dict)
        else:
            outputs = forward_fn(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def bert_setter(model_bert, inputs_dict, hidden_states, forward_fn=None):

    hidden_states_ = []

    def get_hook(i):
        def hook(module, inputs, outputs=None):
            if i == 0:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return hidden_states[i]
                else:
                    hidden_states_.append(outputs)

            elif 1 <= i <= len(model_bert.encoder.layer):
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + inputs[1:]
                else:
                    hidden_states_.append(inputs[0])

            elif i == len(model_bert.encoder.layer) + 1:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + outputs[1:]
                else:
                    hidden_states_.append(outputs[0])

        return hook

    handles = (
        [model_bert.embeddings.word_embeddings.register_forward_hook(get_hook(0))]
        + [
            layer.register_forward_pre_hook(get_hook(i + 1))
            for i, layer in enumerate(model_bert.encoder.layer)
        ]
        + [
            model_bert.encoder.layer[-1].register_forward_hook(
                get_hook(len(model_bert.encoder.layer) + 1)
            )
        ]
    )

    try:
        if forward_fn is None:
            outputs = model_bert(**inputs_dict)
        else:
            outputs = forward_fn(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)

