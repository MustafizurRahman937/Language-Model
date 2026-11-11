import torch
import torch.nn as nn


# --------------------------------------------------
# Lightweight ALBERT-Inspired Prototype Model
# --------------------------------------------------

class SimpleSLM(nn.Module):

    def __init__(
        self,
        vocab_size=30522,
        embedding_dim=64,
        num_classes=2,
        max_length=64
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # --------------------------------------------------
        # Positional embedding
        # Helps the model understand word order
        # --------------------------------------------------

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

        # --------------------------------------------------
        # Lightweight Transformer block
        # --------------------------------------------------

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        # --------------------------------------------------
        # Shared Transformer block
        #
        # The same block is reused multiple times.
        # This is inspired by parameter sharing in ALBERT.
        # --------------------------------------------------

        self.num_shared_layers = 2

        # --------------------------------------------------
        # Classification network
        #
        # Mean pooling + Max pooling
        # gives 128 features (64 + 64)
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                64
            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                32
            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                num_classes
            )
        )


    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    def forward(
        self,
        input_ids,
        attention_mask=None
    ):

        batch_size, sequence_length = input_ids.shape

        # --------------------------------------------------
        # Make sure sequence does not exceed max length
        # --------------------------------------------------

        if sequence_length > self.max_length:

            input_ids = input_ids[:, :self.max_length]

            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]

            sequence_length = self.max_length


        # --------------------------------------------------
        # Token embeddings
        # --------------------------------------------------

        token_embeddings = self.embedding(
            input_ids
        )


        # --------------------------------------------------
        # Position embeddings
        # --------------------------------------------------

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length
        )

        position_embeddings = self.position_embedding(
            positions
        )


        # --------------------------------------------------
        # Combine token + position information
        # --------------------------------------------------

        embeddings = (
            token_embeddings
            +
            position_embeddings
        )


        # --------------------------------------------------
        # Transformer attention
        # --------------------------------------------------

        if attention_mask is not None:

            # Transformer expects:
            # True  = ignore this token
            # False = use this token

            padding_mask = (
                attention_mask == 0
            )

            embeddings = self.transformer_layer(
                embeddings,
                src_key_padding_mask=padding_mask
            )

        else:

            embeddings = self.transformer_layer(
                embeddings
            )


        # --------------------------------------------------
        # Parameter sharing
        #
        # Reuse the SAME Transformer block.
        # This keeps the model lightweight.
        # --------------------------------------------------

        for _ in range(
            self.num_shared_layers - 1
        ):

            if attention_mask is not None:

                embeddings = self.transformer_layer(
                    embeddings,
                    src_key_padding_mask=padding_mask
                )

            else:

                embeddings = self.transformer_layer(
                    embeddings
                )


        # --------------------------------------------------
        # Mask padding tokens
        # --------------------------------------------------

        if attention_mask is not None:

            mask = attention_mask.unsqueeze(-1).float()

            # ------------------------------
            # Mean pooling
            # ------------------------------

            masked_embeddings = (
                embeddings * mask
            )

            summed_embeddings = (
                masked_embeddings.sum(dim=1)
            )

            token_count = (
                mask.sum(dim=1)
                .clamp(min=1)
            )

            mean_pool = (
                summed_embeddings
                /
                token_count
            )

            # ------------------------------
            # Max pooling
            # ------------------------------

            max_embeddings = embeddings.masked_fill(
                mask == 0,
                -1e9
            )

            max_pool = max_embeddings.max(
                dim=1
            ).values

        else:

            mean_pool = embeddings.mean(
                dim=1
            )

            max_pool = embeddings.max(
                dim=1
            ).values


        # --------------------------------------------------
        # Combine mean + max information
        # --------------------------------------------------

        pooled_embeddings = torch.cat(
            [
                mean_pool,
                max_pool
            ],
            dim=1
        )


        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        output = self.classifier(
            pooled_embeddings
        )


        return output