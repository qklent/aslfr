# Google - ASL Fingerspelling Recognition

This repository contains the codebase to reproduce solution to the Google - ASL Fingerspelling Recognition competition on kaggle.

competition link: https://www.kaggle.com/competitions/asl-fingerspelling/overview

## Model
My solution involved a combination of a 1D CNN and a Transformer, trained from scratch. The 1D CNN model employed depthwise convolution and causal padding. The Transformer used BatchNorm + Swish instead of the typical LayerNorm + GELU, due to slightly(negligible) lighter inference with the same accuracy

## Regularization
  1) Dropout in cnn and transformer with drop rate = 0.4
  2) Late dropout on last classifier layer with drop rate = 0.8
  3) Drop Path drop rate = 0.2


## Augmentation
  1) Horizontal Flip
  2) Random Affine Transformation
  3) Random Masking

## Score
Levenshtein distance:

    cv 0.8
  
    lb 0.69

## Tried but not worked
  1) reversing frames
  2) mask ctc https://arxiv.org/abs/2005.08700
  3) transformer for machine translation like in paper Attention is all you need but instead of embedding layer for encoder i used 1D CNN to squish sequence of frames (since 1 char can be represented by 2 or more frames)
  4) transformer like in (3) but with convolution attention in encoder
  

