# Copyright (c) Markus Hiller and Rongkai Ma -- 2022
# All rights reserved.
#
# Dual Reconstruction Module for Few-Shot Learning
# Based on attention-based feature reconstruction

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualReconstruction(nn.Module):
    """
    Dual Reconstruction Module implementing Support→Query and Query→Support reconstruction
    using attention-based feature projection (cosine similarity + softmax)
    """
    def __init__(self, lambda_q=1.0, lambda_s=1.0, lambda_sim=0.5):
        """
        Args:
            lambda_q: Weight for query reconstruction loss
            lambda_s: Weight for support reconstruction loss
            lambda_sim: Weight for similarity combination (between original and reconstructed)
        """
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_s = lambda_s
        self.lambda_sim = lambda_sim
        
    def normalize_features(self, x):
        """
        Normalize features by L2 norm along channel dimension
        
        Args:
            x: (B, L, C) or (B, C, H, W) feature tensor
        Returns:
            normalized: same shape as input
        """
        if x.dim() == 3:
            # (B, L, C) format
            norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=1e-8)
            return x / norm
        elif x.dim() == 4:
            # (B, C, H, W) format
            norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=1e-8)
            return x / norm
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
    
    def compute_reconstruction_weights(self, query_feat, support_feats):
        """
        Compute attention weights for reconstruction using cosine similarity
        
        Args:
            query_feat: (B, L, C) query feature
            support_feats: (K, L, C) support features from K samples
        Returns:
            attention_weights: (B, K, L, L) attention weights
        """
        B, L_q, C = query_feat.shape
        K, L_s, C_s = support_feats.shape
        
        assert C == C_s, "Channel dimensions must match"
        
        # Normalize
        query_norm = self.normalize_features(query_feat)  # (B, L_q, C)
        support_norm = self.normalize_features(support_feats)  # (K, L_s, C)
        
        # Compute similarity matrix for each support sample
        # query_norm: (B, L_q, C), support_norm: (K, L_s, C)
        # We want (B, K, L_q, L_s) similarity matrix
        
        # Expand dimensions for broadcasting
        query_expanded = query_norm.unsqueeze(1)  # (B, 1, L_q, C)
        support_expanded = support_norm.unsqueeze(0)  # (1, K, L_s, C)
        
        # Compute cosine similarity: dot product of normalized features
        # (B, K, L_q, L_s)
        similarity = torch.matmul(query_expanded, support_expanded.transpose(-2, -1))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(similarity, dim=-1)  # (B, K, L_q, L_s)
        
        return attention_weights
    
    def reconstruct_query_from_support(self, query_feat, support_feats):
        """
        Reconstruct query features from support features (Support → Query)
        
        Args:
            query_feat: (B, L, C) query feature
            support_feats: (K, L, C) support features
        Returns:
            reconstructed_query: (B, L, C) reconstructed query
            attention_weights: (B, K, L, L) attention weights
        """
        # Compute attention weights
        attn_weights = self.compute_reconstruction_weights(query_feat, support_feats)
        # attn_weights: (B, K, L_q, L_s)
        
        B, K, L_q, L_s = attn_weights.shape
        C = query_feat.shape[-1]
        
        # Weighted sum over support samples and spatial locations
        # attn_weights: (B, K, L_q, L_s)
        # support_feats: (K, L_s, C)
        # We want to sum over K and L_s dimensions
        
        # Efficient computation: sum over K and L_s
        # For each query sample b, sum over support samples k and spatial locations l_s
        # reconstructed[b, l_q, c] = sum_k sum_l_s (attn_weights[b, k, l_q, l_s] * support_feats[k, l_s, c])
        reconstructed = torch.zeros(B, L_q, C, device=query_feat.device, dtype=query_feat.dtype)
        for k in range(K):
            # attn_weights[:, k, :, :]: (B, L_q, L_s)
            # support_feats[k, :, :]: (L_s, C)
            # Result: (B, L_q, C)
            reconstructed += torch.matmul(attn_weights[:, k, :, :], support_feats[k, :, :])
        
        return reconstructed, attn_weights
    
    def reconstruct_support_from_query(self, support_feat, query_feats):
        """
        Reconstruct support features from query features (Query → Support)
        
        Args:
            support_feat: (K, L, C) support feature
            query_feats: (B, L, C) query features
        Returns:
            reconstructed_support: (K, L, C) reconstructed support
            attention_weights: (K, B, L, L) attention weights
        """
        K, L_s, C = support_feat.shape
        B, L_q, C_q = query_feats.shape
        
        assert C == C_q, "Channel dimensions must match"
        
        # Normalize
        support_norm = self.normalize_features(support_feat)  # (K, L_s, C)
        query_norm = self.normalize_features(query_feats)  # (B, L_q, C)
        
        # Compute similarity: (K, B, L_s, L_q)
        support_expanded = support_norm.unsqueeze(1)  # (K, 1, L_s, C)
        query_expanded = query_norm.unsqueeze(0)  # (1, B, L_q, C)
        
        similarity = torch.matmul(support_expanded, query_expanded.transpose(-2, -1))
        attn_weights = F.softmax(similarity, dim=-1)  # (K, B, L_s, L_q)
        
        # Efficient reconstruction: sum over B and L_q
        # For each support sample k, sum over query samples b and spatial locations l_q
        # reconstructed[k, l_s, c] = sum_b sum_l_q (attn_weights[k, b, l_s, l_q] * query_feats[b, l_q, c])
        reconstructed = torch.zeros(K, L_s, C, device=support_feat.device, dtype=support_feat.dtype)
        for b in range(B):
            # attn_weights[:, b, :, :]: (K, L_s, L_q)
            # query_feats[b, :, :]: (L_q, C)
            # Result: (K, L_s, C)
            reconstructed += torch.matmul(attn_weights[:, b, :, :], query_feats[b, :, :])
        
        return reconstructed, attn_weights
    
    def compute_reconstruction_loss(self, original, reconstructed):
        """
        Compute cosine distance loss between original and reconstructed features
        
        Args:
            original: (B, L, C) or (B, C, H, W) original features
            reconstructed: (B, L, C) or (B, C, H, W) reconstructed features
        Returns:
            loss: scalar reconstruction loss
        """
        # Flatten if needed
        if original.dim() == 4:
            B, C, H, W = original.shape
            original = original.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
            reconstructed = reconstructed.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        # Normalize
        original_norm = self.normalize_features(original)
        reconstructed_norm = self.normalize_features(reconstructed)
        
        # Cosine similarity
        cosine_sim = (original_norm * reconstructed_norm).sum(dim=-1).mean()
        
        # Cosine distance loss: 1 - cosine_similarity
        loss = 1.0 - cosine_sim
        
        return loss
    
    def forward(self, query_features, support_features):
        """
        Perform dual reconstruction: Support→Query and Query→Support
        
        Args:
            query_features: (B, L, C) query features after DRFF
            support_features: (K, L, C) support features after DRFF
        Returns:
            results: dict containing:
                - reconstructed_query: (B, L, C)
                - reconstructed_support: (K, L, C)
                - query_reconstruction_loss: scalar
                - support_reconstruction_loss: scalar
                - total_reconstruction_loss: scalar
                - query_attn_weights: (B, K, L, L)
                - support_attn_weights: (K, B, L, L)
        """
        # Support → Query Reconstruction
        reconstructed_query, query_attn = self.reconstruct_query_from_support(
            query_features, support_features)
        
        # Query → Support Reconstruction
        reconstructed_support, support_attn = self.reconstruct_support_from_query(
            support_features, query_features)
        
        # Compute losses
        query_loss = self.compute_reconstruction_loss(query_features, reconstructed_query)
        support_loss = self.compute_reconstruction_loss(support_features, reconstructed_support)
        
        total_loss = self.lambda_q * query_loss + self.lambda_s * support_loss
        
        return {
            'reconstructed_query': reconstructed_query,
            'reconstructed_support': reconstructed_support,
            'query_reconstruction_loss': query_loss,
            'support_reconstruction_loss': support_loss,
            'total_reconstruction_loss': total_loss,
            'query_attn_weights': query_attn,
            'support_attn_weights': support_attn
        }
    
    def compute_classification_similarity(self, query_orig, query_recon, support_orig, support_recon):
        """
        Compute classification similarity from original and reconstructed features
        
        Args:
            query_orig: (B, L, C) original query
            query_recon: (B, L, C) reconstructed query
            support_orig: (K, L, C) original support
            support_recon: (K, L, C) reconstructed support
        Returns:
            similarity: (B, K) similarity scores
        """
        # Normalize
        q_orig_norm = self.normalize_features(query_orig)  # (B, L, C)
        q_recon_norm = self.normalize_features(query_recon)  # (B, L, C)
        s_orig_norm = self.normalize_features(support_orig)  # (K, L, C)
        s_recon_norm = self.normalize_features(support_recon)  # (K, L, C)
        
        # Compute similarities
        # sim1 = cos(q_orig, s_recon)
        # Average over spatial dimension
        q_orig_avg = q_orig_norm.mean(dim=1)  # (B, C)
        s_recon_avg = s_recon_norm.mean(dim=1)  # (K, C)
        sim1 = torch.matmul(q_orig_avg, s_recon_avg.transpose(0, 1))  # (B, K)
        
        # sim2 = cos(q_recon, s_orig)
        q_recon_avg = q_recon_norm.mean(dim=1)  # (B, C)
        s_orig_avg = s_orig_norm.mean(dim=1)  # (K, C)
        sim2 = torch.matmul(q_recon_avg, s_orig_avg.transpose(0, 1))  # (B, K)
        
        # Weighted combination
        similarity = self.lambda_sim * sim1 + (1 - self.lambda_sim) * sim2
        
        return similarity

