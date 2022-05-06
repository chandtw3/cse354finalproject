'''
author: Sounak Mondal
'''

# std lib imports
from msilib import sequence
from typing import Dict

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

class SequenceToVector(nn.Module):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.dropout = dropout
        self.device = device
        self.layers=nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(str(len(self.layers)), nn.Linear(input_dim,input_dim))
            if i < num_layers - 1:
                self.layers.add_module(str(len(self.layers)), nn.ReLU())
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        sequence_mask = torch.unsqueeze(sequence_mask,2).expand(vector_sequence.size()[0],vector_sequence.size()[1],vector_sequence.size()[2])
        vector_sequence = torch.mul(vector_sequence, sequence_mask)
        counts_mask = vector_sequence.count_nonzero(1)
        if training:
            bernoulli_dropout = torch.distributions.Bernoulli(self.dropout).sample((vector_sequence.shape[1],))
            vector_sequence = vector_sequence[:, bernoulli_dropout==1]
        vector_sequence = torch.div(torch.sum(vector_sequence, 1), counts_mask)
        layer_representations=None
        for i in range(len(self.layers)):
            vector_sequence = self.layers[i](vector_sequence)
            if i % 2 == 0:
                if layer_representations == None:
                    layer_representations = torch.unsqueeze(vector_sequence,0)
                else:
                    layer_representations = torch.cat((layer_representations, torch.unsqueeze(vector_sequence,0)),0)

        combined_vector = vector_sequence
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.device = device
        self.hidden_size = input_dim
        self.gru = nn.GRU(input_dim, input_dim, num_layers)
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        vector_sequence = torch.nn.utils.rnn.pack_padded_sequence(vector_sequence, torch.count_nonzero(sequence_mask, 1), True, False)
        output, hidden = self.gru(vector_sequence)
        combined_vector = torch.select(hidden, 0 ,-1)
        layer_representations = hidden
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
