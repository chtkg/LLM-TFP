Overview

The goal of this project is to apply large language models to the problem of traffic flow prediction. LLM-TFP is configured to handle time series data in a manner similar to 
sequences in natural language processing, adapting the BERT architecture to capture both spatial and temporal dependencies in traffic data.

Datasets

We evaluate the model using two public traffic datasets:
METR-LA: Traffic data from 207 sensors in Los Angeles, spanning from March to June 2012.
PEMS-BAY: Data from 325 sensors in the California Bay Area, collected from January to May 2017.

Model Architecture

The LLM-TFP model builds on a BERT-based architecture, adapted for time series data. Key settings include:
Time Dimension (T_d): Set to 288 for 5-minute intervals across 24 hours.
Optimizer: Ranger21 with a learning rate of 0.0001.
Training: 25 epochs, batch size of 32.
See model/LLM_TFP.py for implementation details.

Installation

Prerequisites

Python 3.8+
PyTorch 1.8+
CUDA (optional, for GPU support)

Clone the repository:
git clone https://github.com/chtkg/LLM-TFP.git

cd LLM-TFP
